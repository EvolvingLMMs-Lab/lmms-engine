from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    Qwen3Config,
    Qwen3ForCausalLM,
    Qwen3Model,
)
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast

from lmms_engine.models.utils import sde_step_with_logprob 
from ..blip3o_arch import blip3oMetaForCausalLM, blip3oMetaModel, rank0_print
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

import numpy as np
from tqdm import tqdm

import PIL

def numpy_to_pil(images: np.ndarray):
    """
    Convert a NumPy array of shape (batch, height, width, channels) to a list of PIL Images.
    """
    pil_images = []
    for img in images:
        img_uint8 = (img * 255).round().astype("uint8")
        if img_uint8.shape[2] == 1:
            img_uint8 = img_uint8[..., 0]
        pil_images.append(PIL.Image.fromarray(img_uint8))
    return pil_images



class blip3oQwenConfig(Qwen3Config):
    model_type = "blip3o_qwen"

class blip3oQwenModel(blip3oMetaModel, Qwen3Model):
    config_class = blip3oQwenConfig

    def __init__(self, config: Qwen3Config):
        super(blip3oQwenModel, self).__init__(config)

class blip3oQwenForCausalLM(Qwen3ForCausalLM, blip3oMetaForCausalLM):
    config_class = blip3oQwenConfig
    _supports_sdpa = True
    _supports_flash_attn = True
    supports_gradient_checkpointing = True

    def __init__(self, config):
        Qwen3ForCausalLM.__init__(self, config)
        config.model_type = "blip3o_qwen"
        config.rope_scaling = None

        self.model = blip3oQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def get_sigmas(self, timesteps, device, n_dim=4, dtype=torch.float32):
        sigmas = self.model.noise_scheduler.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = self.model.noise_scheduler.timesteps.to(device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def mask_drop(self, latents, drop_prob=0.1):
        if drop_prob <= 0:
            return latents
        mask = torch.bernoulli(torch.zeros(latents.shape[0], device=latents.device, dtype=latents.dtype) + drop_prob)
        while len(mask.shape) < len(latents.shape):
            mask = mask.unsqueeze(-1)
        mask = 1 - mask  # need to flip 0 <-> 1
        return latents * mask


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        target_images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        dpo_forward: Optional[bool] = False,
        cache_position=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:


        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = 0
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if target_images is not None:
            vae = self.model.get_sana_vae()
            latents = vae.encode(target_images).latent
            if "shift_factor" in vae.config and vae.config.shift_factor is not None:
                latents = latents - vae.config.shift_factor
            latents = latents * vae.config.scaling_factor
            noise = torch.randn_like(latents, device=latents.device)
            weighting_scheme = "uniform"
            u = compute_density_for_timestep_sampling(
                weighting_scheme=weighting_scheme,
                batch_size=latents.shape[0],
                logit_mean=0.0,
                logit_std=1.0,
                mode_scale=1.29,
            )
            indices = (u * self.model.noise_scheduler.config.num_train_timesteps).long()
            timesteps = self.model.noise_scheduler.timesteps[indices].to(device=latents.device)
            sigmas = self.get_sigmas(timesteps, latents.device, n_dim=latents.ndim, dtype=latents.dtype)
            noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
            
            sana = self.model.get_sana()


            start_pos = (labels == self.config.image_start_tag_id).float().argmax(dim=1)   
            end_pos   = (labels == self.config.image_end_tag_id).float().argmax(dim=1)   

            selected_hidden_states = []                       
            for b in range(hidden_states.size(0)):          
                start = start_pos[b].item() + 1         
                end = end_pos[b].item()      
                hidden_states_filter = hidden_states[b, start:end, :]      
                if hidden_states_filter.size(1) != 730:
                    hidden_states_filter = hidden_states[b, -730:, :]
                selected_hidden_states.append(hidden_states_filter) 

            selected_hidden_states = torch.stack(selected_hidden_states, dim=0)
            diffusion_pred = sana(
                hidden_states=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=self.model.diffusion_connector(self.mask_drop(selected_hidden_states)),
                encoder_attention_mask=None,
            ).sample

            target = noise - latents
            weighting = compute_loss_weighting_for_sd3(weighting_scheme=weighting_scheme, sigmas=sigmas)
            diff_loss = torch.mean(
                (weighting.float() * (diffusion_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                1,
            )
            diff_loss = diff_loss.mean()
            rank0_print(f" Cross-entropy loss {loss}, Diffusion loss {diff_loss} ")
            loss += diff_loss

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes)
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
        return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)

    @torch.no_grad()
    def decode_latents(self, latents, normalize=True, return_tensor=False):
        if self.model.sana_vae is not None:
            latents = latents / self.model.sana_vae.config.scaling_factor
            if "shift_factor" in self.model.sana_vae.config and self.model.sana_vae.config.shift_factor is not None:
                latents = latents + self.model.sana_vae.config.shift_factor
            samples = self.model.sana_vae.decode(latents).sample
        else:
            samples = latents
        if normalize:
            samples = (samples / 2 + 0.5).clamp(0, 1)
        else:
            samples = samples.clamp(-1, 1)
        if return_tensor:
            return samples
        samples = samples.cpu().permute(0, 2, 3, 1).float().numpy()
        samples = numpy_to_pil(samples)
        return samples

    @torch.no_grad()
    def generate_gen_ids(self, input_ids, attention_mask, do_sample=True, temperature=1.0, max_new_tokens=None, **kwargs):
        return super().generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            attention_mask=attention_mask,
            use_cache=True,
            **kwargs
        )

    def get_image_start_tag_id(self):
        return self.config.image_start_tag_id

    def get_scheduler(self):
        return self.model.noise_scheduler

    @torch.no_grad()
    def generate_images(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: Optional[torch.Tensor] = None,
        gen_ids: Optional[torch.Tensor] = None,
        guidance_scale: float = 2.0,
        num_inference_steps: int = 30,
        num_images_per_prompt: int = 1,
        return_tensor=False,
        enable_progress_bar=False,
        use_sde=False,
        **kwargs,
    ):
        if gen_ids is None:
            input_ids = input_ids.repeat(num_images_per_prompt, 1)
            attention_mask = attention_mask.repeat(num_images_per_prompt, 1)
            gen_ids = self.generate_gen_ids(
                input_ids, 
                attention_mask, 
                do_sample=True, 
                temperature=1.0, 
                max_new_tokens=max_new_tokens
            )
        # breakpoint()
        with torch.no_grad():
            outs = self.model(
                input_ids = gen_ids, 
                output_hidden_states = True,
                return_dict = True,
                use_cache=False
            )
        hidden_states = outs.hidden_states[-1]   


        start_pos = (gen_ids == self.config.image_start_tag_id).float().argmax(dim=1)   

        selected_hidden_states = []                       
        for b in range(hidden_states.size(0)):          
            start = start_pos[b].item() + 1         
            # end = end_pos[b].item()              
            selected_hidden_states.append(hidden_states[b, start:, :]) 
        pred_latent = torch.stack(selected_hidden_states, dim=0)
        

        img_hidden_states_null = torch.zeros_like(pred_latent)
        pred_latent = torch.cat([img_hidden_states_null, pred_latent], 0)
        ## sample images from here
        device = next(self.parameters()).device
        # dtype = next(self.parameters()).dtype

        bsz = len(pred_latent) // 2
        # latent_size = self.config.input_size
        latent_size = 32
        latent_channels = self.model.sana.config.in_channels

    
        latents = randn_tensor(
            shape=(bsz * num_images_per_prompt, latent_channels, latent_size, latent_size),
            generator=None,
            device=device,
            dtype=torch.bfloat16,
        )

        # set step values
        if isinstance(self.model.noise_scheduler, FlowMatchEulerDiscreteScheduler):
            sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
            self.model.noise_scheduler.set_timesteps(num_inference_steps, sigmas=sigmas)
        else:
            self.model.noise_scheduler.set_timesteps(num_inference_steps)

        # SDE variables initialization
        ts, log_probs = [], []
        cur_latents = []
        denoised_latents = []

        # pred_latent = torch.cat([pred_latent] * 2)
        # Convert to float32 before saving
        # for t in tqdm(self.model.noise_scheduler.timesteps, desc="Sampling images", disable=not enable_progress_bar):
        for idx, t in enumerate(self.model.noise_scheduler.timesteps):

            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = latent_model_input.to(pred_latent.dtype)
            # print(t, t.unsqueeze(0).expand(latent_model_input.shape[0]))
            # predict noise model_output
            res = self.model.sana(
                hidden_states=latent_model_input,
                encoder_hidden_states=self.model.diffusion_connector(pred_latent),
                timestep=t.unsqueeze(0).expand(latent_model_input.shape[0]).to(latents.device),
                encoder_attention_mask=None
            )
            noise_pred = res.sample
            noise_pred_uncond, noise_pred= noise_pred.chunk(2)

            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)

            # Step: standard vs. SDE
            if use_sde:
                cur_latents.append(latents.unsqueeze(1))
                latents, log_prob, _, _ = sde_step_with_logprob(
                    self.model.noise_scheduler, noise_pred.float(), t.unsqueeze(0), latents.float(),
                )
                log_probs.append(log_prob.unsqueeze(1))
                ts.append(t.unsqueeze(0).repeat(len(log_prob)).unsqueeze(1))
                denoised_latents.append(latents.unsqueeze(1))
                latents = latents.to(dtype=pred_latent.dtype)
            else:
                # compute previous image: x_t -> x_t-1
                latents = self.model.noise_scheduler.step(noise_pred, t, latents).prev_sample

        samples = self.decode_latents(latents.to(self.model.sana_vae.dtype) if self.model.sana_vae is not None else latents, return_tensor=return_tensor)      

        if use_sde:
            # Flatten SDE outputs
            denoised_latents = torch.cat(denoised_latents, dim=1).flatten(0, 1)
            cur_latents = torch.cat(cur_latents, dim=1).flatten(0, 1)
            log_probs = torch.cat(log_probs, dim=1).flatten(0, 1)
            # pred_latents = torch.cat(pred_latents, dim=1).flatten(0, 1)
            ts = torch.cat(ts, dim=1).flatten(0, 1)
            return gen_ids, samples, log_probs, pred_latent, denoised_latents, cur_latents, ts
        else:
            return gen_ids, samples

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs