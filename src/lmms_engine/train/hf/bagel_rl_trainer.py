# Copyright 2025 The HuggingFace Team. All rights reserved.
# Copyright 2025 Fu-Yun Wang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Unified GRPO Trainer for BLIP3o Models (Text-to-Image and Image-to-Image)

This module implements Group Relative Policy Optimization (GRPO) for vision-language
models, supporting both text-to-image generation and image-to-image editing tasks.

"""

import os
import sys
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Union
from datetime import datetime

import torch
import torch.distributed as dist
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AutoProcessor,
    PreTrainedModel,
    is_wandb_available,
)
from copy import deepcopy
from PIL import Image

from .trainer import Trainer
import numpy as np


import transformers
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available
from transformers import CLIPProcessor, CLIPImageProcessor, PreTrainedTokenizerBase

from deepspeed.runtime.zero import GatheredParameters
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode

from accelerate.utils import send_to_device


from contextlib import contextmanager
from lmms_engine.models.utils import sde_step_with_logprob
from lmms_engine.train.registry import TRAINER_REGISTER
import lmms_engine.parallel.process_group_manager as pgm
from lmms_engine.utils.reward_evaluators import (
    recon_reward,
    jpeg_compressibility,
    jpeg_incompressibility,
    clip_similarity,
    sim_direction,
    format_reward,
    RewardEvaluatorClient
)

from diffusers.utils.torch_utils import randn_tensor
from trl.data_utils import maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation

from lmms_engine.models.bagel.qwen2_navit import NaiveCache

if is_wandb_available():
    import wandb

# Type aliases
def compute_log_prob(model_pred: torch.Tensor, 
                     scheduler: FlowMatchEulerDiscreteScheduler,
                     cur_latents: torch.Tensor, 
                     denoised_latents: torch.Tensor, 
                     ts: torch.Tensor):
    """
    Compute log probability and related statistics for SDE step.
    
    Args:
        model_pred: Model prediction output
        scheduler: Flow matching scheduler
        prev_latents: Previous latent state
        pred_latents: Predicted latent state
        ts: Timestep
        
    Returns:
        Tuple of (prev_sample, log_prob, prev_sample_mean, std_dev_t)
    """ 
    prev_sample, log_prob, prev_sample_mean, std_dev_t = sde_step_with_logprob(
        scheduler,
        model_pred.float(),
        ts,
        cur_latents.float(),
        denoised_latents.float(),
    )
    return prev_sample, log_prob, prev_sample_mean, std_dev_t

class BaseBagelRLTrainer(Trainer):
    """
    Base trainer for GRPO with BLIP3o models.
    
    This class provides common functionality for both T2I and I2I variants.
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize BLIP3o GRPO Trainer.
        Args:
            model: Model ID or PreTrainedModel instance
            reward_funcs: Reward function(s) for optimization
            args: Training configuration
            processing_class: Tokenizer/processor
            optimizers: Optimizer and scheduler tuple
            peft_config: PEFT configuration for parameter-efficient training
            max_pixels: Maximum pixels for image processing
            min_pixels: Minimum pixels for image processing
            attn_implementation: Attention implementation type
            task_type: Task type ("t2i" or "i2i")
        """
        # Configuration
        trainer_config = kwargs.get("args", None)
        rl_config = trainer_config.rl_config
        node_addr = rl_config.get("node_addr", "localhost")
        reward_funcs = rl_config.get("reward_funcs", ["aesthetic"])
        self.task_type = rl_config.get("task_type", "t2i")
        self.num_generations = rl_config.get("num_generations", 1)
        self.cfg_text_scale = rl_config.get("cfg_text_scale", 2)
        self.num_inference_steps = rl_config.get("num_inference_steps", 30)
        self.beta = float(rl_config.get("beta", 0.0))
        self.timestep_shift = rl_config.get("timestep_shift", 1.0)
        self.max_prompt_length = rl_config.get("max_prompt_length", 1024)
        self.max_response_length = rl_config.get("max_response_length", 1024)
        self.gen_image_size = rl_config.get("gen_image_size", (512, 512))
        # self._setup_transforms()
        self.scheduler = FlowMatchEulerDiscreteScheduler(shift=self.timestep_shift)
        self.scheduler.set_timesteps(self.num_inference_steps)
        
        # Logging
        self.start_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
        self.log_dir = os.path.join(trainer_config.output_dir, "training_samples", self.start_time)
        os.makedirs(self.log_dir, exist_ok=True)
        self._metrics = defaultdict(list)
        
        # Reward functions
        SCORER_URLS = {
            "aesthetic": "http://" + node_addr + ":18080/",
            "image_reward": "http://" + node_addr + ":18081/",
            "ocr": "http://" + node_addr + ":18082/",
            "pickscore": "http://" + node_addr + ":18083/",
            "deqa": "http://" + node_addr + ":18084/",
            "gen_eval": "http://" + node_addr + ":18085/",
            "unifiedreward_sglang": "http://" + node_addr + ":18086/", 
            "hps": "http://" + node_addr + ":18087/", 
        }
        self.reward_client = RewardEvaluatorClient(scorer_urls=SCORER_URLS)
        self.reward_funcs_registry = {
            "recon": recon_reward,
            "jpeg_compressibility": jpeg_compressibility,
            "jpeg_incompressibility": jpeg_incompressibility,
            "pickscore": self.reward_client.pickscore,
            "deqa": self.reward_client.deqa,
            "gen_eval": self.reward_client.gen_eval,
            "unifiedreward_sglang": self.reward_client.unifiedreward_sglang,
            "ocr": self.reward_client.ocr,
            "image_reward": self.reward_client.image_reward,
            "aesthetic": self.reward_client.aesthetic,
            "hps": self.reward_client.hps,
            "clip_sim": clip_similarity,
            "sim_direction": sim_direction,
            "format": format_reward,
        }
        self.reward_processing_registry = {
            "recon": T.Compose([T.ToTensor()]),
            "jpeg_compressibility": None,
            "jpeg_incompressibility": None,
            "pickscore": None,
            "deqa": None,
            "gen_eval": None,
            "unifiedreward_sglang": None,
            "ocr": None,
            "image_reward": None,
            "aesthetic": None,
            "hps": None,
            "clip_sim": CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14"),
            "sim_direction": CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14"),
            "format": None,
        }
        self.reward_funcs = [
            (
                func, 
                self.reward_processing_registry[func], 
                self.reward_funcs_registry[func]
            ) for func in reward_funcs
        ]
        self.ref_model = None
        super().__init__(*args, **kwargs)
    
    def _set_signature_columns_if_needed(self):
        """Set required dataset columns."""
        if self._signature_columns is None:
            self._signature_columns = ["input_ids", "caption"]
    
    def _prepare_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Skip automatic tensor conversion."""
        return inputs

    def _prepare_generation_inputs_for_fsdp2(self, inputs: Dict[str, Any], model: PreTrainedModel) -> Dict[str, Any]:
        """
        Convert regular tensors to DTensors with Replicate placement for FSDP2.

        This is necessary when calling custom model methods (not forward()) with FSDP2,
        as FSDP2's automatic input conversion only applies to the standard forward() method.

        IMPORTANT: We must use the EXACT same DeviceMesh instance as the model parameters,
        otherwise PyTorch will raise "cross-mesh operation" errors.

        Args:
            inputs: Dictionary of inputs that may contain tensors
            model: The model to get the device_mesh from

        Returns:
            Dictionary with tensors converted to replicated DTensors
        """
        # Get device_mesh from model parameters
        # We MUST use the same device_mesh instance as the model parameters
        device_mesh = None
        for param in model.parameters():
            if isinstance(param, DTensor):
                device_mesh = param.device_mesh
                break

        # If no DTensor parameters found, inputs don't need conversion
        if device_mesh is None:
            return inputs

        # Convert each tensor in the inputs to a replicated DTensor
        converted_inputs = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor) and not isinstance(value, DTensor):
                # Convert regular tensor to replicated DTensor
                # Replicate means each rank has a full copy of the tensor
                converted_inputs[key] = distribute_tensor(
                    value,
                    device_mesh,
                    [Replicate()]
                )
            else:
                # Keep non-tensor values or already-DTensor values as-is
                converted_inputs[key] = value

        return converted_inputs


    def _compute_rewards(self, inputs: List[Dict], images: List[Any], completions: Optional[List[str]] = None) -> torch.Tensor:
        """
        Compute rewards from all reward functions.
        
        Args:
            inputs: Input batch
            images: Generated images
            
        Returns:
            Tensor of shape (batch_size * num_generations,) with rewards
        """
        device = self.accelerator.device
        rewards_per_func = torch.zeros(len(images), len(self.reward_funcs), device=device)
        
        # Extract metadata
        captions = [ex.get("caption", ex.get("target_caption", "")) for ex in inputs]
        
        for i, (func_name, _, reward_func) in enumerate(self.reward_funcs):
            if func_name == "jpeg_compressibility" or func_name == "jpeg_incompressibility":
                rewards_per_func[:, i] = reward_func(images)
            elif func_name in ["pickscore", "hps", "deqa", "image_reward", "aesthetic"]:
                scores = reward_func(
                    images, 
                    [cap for cap in captions for _ in range(self.num_generations)]
                )["scores"]
                rewards_per_func[:, i] = torch.tensor(scores).to(device)
            elif func_name == "format":
                assert completions is not None, "completions must be provided for format reward"
                rewards_per_func[:, i] = torch.tensor(reward_func(completions)).to(device)
            elif func_name == "gen_eval":
                meta_files = [ex.get("metadata") for ex in inputs]
                meta_input = {"meta_datas": [m for m in meta_files for _ in range(self.num_generations)]}
                scores = reward_func(
                    images,
                    [cap for cap in captions for _ in range(self.num_generations)],
                    meta_input
                )["scores"]
                rewards_per_func[:, i] = torch.tensor(scores).to(device)
        
        # Aggregate rewards (can be customized)
        return rewards_per_func.sum(dim=1), rewards_per_func

    def _log_step(self, images, advantages, completions):
        global_step = self.state.global_step
        
        if not global_step % 5 == 0:
            return 
    
        device_id = str(self.model.device).replace(":", "")
        
        log_dir = self.log_dir
        
        text_content = f"Prompt: {prompts_text[0]}"
        
        if completions is not None:
            for idx in range (self.num_generations):
                text_content += f"\nCompletion {idx}: {completions[idx]}"
            
        if os.path.exists(os.path.join(log_dir, f"step_{global_step}_{device_id}.txt")):
            return 
        # with open(os.path.join(log_dir, f"step_{global_step}_{device_id}.txt"), "w", encoding="utf-8") as f:
        #     f.write(text_content)
            
        for idx in range(self.num_generations):
            rev_img = images[idx]
            rev_img_pil = rev_img 
            advantage = advantages[idx]
            rev_img_pil.save(os.path.join(log_dir, f"step_{global_step}_{device_id}_{advantage.item()}_{idx}.jpg"))

    def compute_loss(
        self, 
        model: PreTrainedModel, 
        inputs: List[Dict[str, Any]], 
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None
    ):
        """
        Compute GRPO loss.
        
        Args:
            model: Policy model
            inputs: Batch of inputs
            return_outputs: Whether to return outputs (not supported)
            num_items_in_batch: Number of items in batch
            
        Returns:
            Loss tensor
        """
        if return_outputs:
            raise ValueError("GRPOTrainer does not support returning outputs")
        
        # This method should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement compute_loss")
    
    def log(self, logs: Dict[str, float], start_time: Optional[float] = None):
        """Log metrics with custom tracking."""
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}
        logs = {**logs, **metrics}
        
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:
            super().log(logs)
        
        self._metrics.clear()

    def _update_old_model(self, model):
        if is_deepspeed_zero3_enabled():
            from deepspeed.runtime.zero import GatheredParameters
            model_params = list(model.parameters())
            old_params = list(self.old_model.parameters())
            with GatheredParameters(model_params + old_params, modifier_rank=0):
                if torch.distributed.get_rank() == 0:
                    for old_p, new_p in zip(old_params, model_params):
                        old_p.data.copy_(new_p.data)
        else:
            self.old_model.load_state_dict(model.state_dict())

@TRAINER_REGISTER.register("bagel_t2i_grpo_trainer")
class T2IGRPOTrainer(BaseBagelRLTrainer):
    """GRPO Trainer for Text-to-Image generation."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_fsdp_forward_method_flag = False
        
        
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute T2I GRPO loss."""
        if return_outputs:
            raise ValueError("GRPOTrainer does not support returning outputs")
        device = self.accelerator.device
        # device = model.device
        if not self.register_fsdp_forward_method_flag:
            from torch.distributed.fsdp import FSDPModule, register_fsdp_forward_method
            register_fsdp_forward_method(model, "forward_cache_update_text")
            register_fsdp_forward_method(model, "generate_image")
            self.register_fsdp_forward_method_flag = True

        tokenizer = self.train_dataset.processor.tokenizer
        cfg_text_scale = self.cfg_text_scale
        model.eval()
        unwrapped_model = model
        with torch.no_grad():
            new_token_ids = {
                "bos_token_id": tokenizer.convert_tokens_to_ids("<|im_start|>"),
                "eos_token_id": tokenizer.convert_tokens_to_ids("<|im_end|>"),
                "start_of_image": tokenizer.convert_tokens_to_ids("<|vision_start|>"),
                "end_of_image": tokenizer.convert_tokens_to_ids("<|vision_end|>"),
            }
            prompts = inputs["prompts"] * self.num_generations
            bs = len(prompts)
            p = unwrapped_model.latent_patch_size
            num_latent_downsample = unwrapped_model.latent_downsample
            C = unwrapped_model.latent_channel
            H, W = self.gen_image_size
            image_sizes = [(H, W)] * bs
            
            assert H % (num_latent_downsample) == 0, "gen_image_height must be divisible by latent_patch_size * latent_downsample"
            assert W % (num_latent_downsample) == 0, "gen_image_width must be divisible by latent_patch_size * latent_downsample"
            gen_patchified_vae_latent_shapes = [(H // num_latent_downsample, W // num_latent_downsample)] * self.num_generations
            past_key_values = NaiveCache(unwrapped_model.config.llm_config.num_hidden_layers)
            curr_kvlens = [0] * bs  # 当前 KV cache 长度
            curr_rope = [0] * bs    # 当前 RoPE 位置
            generation_input, curr_kvlens, curr_rope = unwrapped_model.prepare_prompts(
                curr_kvlens=curr_kvlens,
                curr_rope=curr_rope,
                prompts=prompts,
                tokenizer=tokenizer,
                new_token_ids=new_token_ids,
            )
            for k, v in generation_input.items():
                if isinstance(v, torch.Tensor):
                    generation_input[k] = send_to_device(v, device)

            past_key_values = unwrapped_model.forward_cache_update_text(
                past_key_values=past_key_values,
                **generation_input
            )

            generation_input = unwrapped_model.prepare_vae_latent(
                curr_kvlens=curr_kvlens,
                curr_rope=curr_rope,
                image_sizes=image_sizes,  # [(H, W)]
                new_token_ids=new_token_ids,
            )
            
            for k, v in generation_input.items():
                if isinstance(v, torch.Tensor):
                    generation_input[k] = send_to_device(v, device)

            cfg_text_past_key_values = None

            if cfg_text_scale > 1.0:
                cfg_text_past_key_values = NaiveCache(unwrapped_model.config.llm_config.num_hidden_layers)
                cfg_curr_kvlens = [0] * bs
                cfg_curr_rope = [0] * bs
                cfg_generation_input, cfg_curr_kvlens, cfg_curr_rope = unwrapped_model.prepare_prompts(
                    curr_kvlens=cfg_curr_kvlens,
                    curr_rope=cfg_curr_rope,
                    prompts=[""] * bs,
                    tokenizer=tokenizer,
                    new_token_ids=new_token_ids,
                )
                cfg_text_past_key_values = unwrapped_model.forward_cache_update_text(
                    cfg_text_past_key_values,
                    **cfg_generation_input
                )
                cfg_generation_input = unwrapped_model.prepare_vae_latent_cfg(
                    curr_kvlens=cfg_curr_kvlens,
                    curr_rope=cfg_curr_rope,
                    image_sizes=image_sizes,
                )
                for k, v in cfg_generation_input.items():
                    if isinstance(v, torch.Tensor):
                        cfg_generation_input[k] = send_to_device(v, device)
            # print(generation_input["packed_text_ids"].shape, generation_input["packed_text_indexes"].shape, generation_input["packed_init_noises"].shape, generation_input["packed_vae_position_ids"].shape, generation_input["packed_vae_token_indexes"].shape)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                unpacked_latent, log_probs, denoised_latents, cur_latents, ts = unwrapped_model.generate_image(
                    packed_text_ids=generation_input["packed_text_ids"],
                    packed_text_indexes=generation_input["packed_text_indexes"],
                    packed_init_noises=generation_input["packed_init_noises"],
                    packed_vae_position_ids=generation_input["packed_vae_position_ids"],
                    packed_vae_token_indexes=generation_input["packed_vae_token_indexes"],
                    packed_seqlens=generation_input["packed_seqlens"],
                    packed_position_ids=generation_input["packed_position_ids"],
                    packed_indexes=generation_input["packed_indexes"],
                    past_key_values=past_key_values,
                    key_values_lens=generation_input["key_values_lens"],
                    packed_key_value_indexes=generation_input["packed_key_value_indexes"],
                    num_timesteps=self.num_inference_steps,
                    cfg_text_scale=cfg_text_scale,
                    cfg_text_packed_query_indexes=cfg_generation_input["cfg_packed_query_indexes"],
                    cfg_text_packed_position_ids=cfg_generation_input["cfg_packed_position_ids"],
                    cfg_text_past_key_values=cfg_text_past_key_values,
                    cfg_text_key_values_lens=cfg_generation_input["cfg_key_values_lens"],
                    cfg_text_packed_key_value_indexes=cfg_generation_input["cfg_packed_key_value_indexes"],
                    use_sde=True,
                    timestep_shift=self.timestep_shift,
                    sde_scheduler=self.scheduler,
                )

                images = []
                # print(len(unpacked_latent), unpacked_latent[0].shape,len(gen_patchified_vae_latent_shapes))
                for packed_latent, (h, w) in zip(unpacked_latent, gen_patchified_vae_latent_shapes):
                    # 6.1 Unpatchify
                    latent = packed_latent.reshape(h, w, p, p, C)
                    latent = torch.einsum("hwpqc->chpwq", latent)
                    latent = latent.reshape(C, h * p, w * p).unsqueeze(0)
                    # 6.2 VAE Decode
                    image = unwrapped_model.vae_model.decode(latent)
                    # 6.3 后处理
                    image = image.squeeze(0).float()
                    image = image.clamp(0, 1)
                    image = (image * 255).byte()
                    image = image.permute(1, 2, 0).cpu().numpy()
                    image = Image.fromarray(image)
                    images.append(image)

        # Compute rewards and advantages
        inputs_for_rewards = [{"caption": prompt} for prompt in inputs["prompts"]]
        rewards, rewards_per_func = self._compute_rewards(inputs_for_rewards, images)
        reshaped_rewards = rewards.view(-1, self.num_generations)
        mean_rewards = reshaped_rewards.mean(dim=1).repeat_interleave(self.num_generations)
        if self.num_generations > 1:
            std_rewards = reshaped_rewards.std(dim=1).repeat_interleave(self.num_generations)
        else:
            raise ValueError("num_generations must be greater than 1 for grpo training")
        advantages = (rewards - mean_rewards) / (std_rewards + 1e-4)
        advantages = torch.clamp(advantages, -5, 5)
        
        self._log_step(images, advantages, None)

        policy_noise_preds = self._compute_diffusion_pred(
            model,
            diffusion_latents=diffusion_latents,
            traj_cur_latents=traj_latents,
            ts=ts,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
            num_images_per_prompt=self.num_generations
        )

        if self.ref_model is not None:
            with torch.no_grad():
                ref_noise_preds = self._compute_diffusion_pred(
                    self.ref_model,
                    diffusion_latents=diffusion_latents,
                    traj_cur_latents=traj_latents,
                    ts=ts,
                    guidance_scale=self.guidance_scale,
                    num_inference_steps=self.num_inference_steps,
                    num_images_per_prompt=self.num_generations
                )
        
        # Compute log probs and KL
        _, policy_log_probs, policy_mean, policy_std = compute_log_prob(
            policy_noise_preds, model.get_scheduler(), traj_latents, traj_denoised_latents, ts
        )
        _, _, ref_mean, ref_std = compute_log_prob(
            ref_noise_preds, model.get_scheduler(), traj_latents, traj_denoised_latents, ts
        )
        
        kl = (policy_mean - ref_mean)**2 / (2 * policy_std**2)
        kl = kl.mean(dim=tuple(range(1, kl.ndim)))
        
        # GRPO loss
        advantages_steps = advantages.repeat_interleave(
            self.num_inference_steps, dim=0
        )
        ratio = torch.exp(policy_log_probs - traj_log_probs)
        unclipped_loss = -advantages_steps * ratio
        clipped_loss = -advantages_steps * torch.clamp(ratio, 1.0 - 1e-4, 1.0 + 1e-4)
        policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
        
        loss = policy_loss + self.beta * kl.mean()
        
        # Logging
        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(kl).mean().item())
        
        for i, (func_name, _, _) in enumerate(self.reward_funcs):
            self._metrics[f"reward/{func_name}"].append(
                self.accelerator.gather_for_metrics(
                    rewards_per_func[:, i]
                ).mean().item())
                
        return loss
    
    def _compute_diffusion_pred(
        self, 
        model_to_use: PreTrainedModel,
        traj_cur_latents: torch.Tensor,
        ts: Optional[torch.Tensor] = None,
        guidance_scale: float = 2.0,
        num_inference_steps: int = 30,
        num_images_per_prompt: int = 1,
        **kwargs
    ):
        latent_model_input = torch.cat([traj_cur_latents] * 2)
        latent_model_input = latent_model_input.to(diffusion_latents.dtype)

        model_base = model_to_use.get_model()
        img_hidden_states = model_base.diffusion_connector(diffusion_latents)

        img_hidden_states = img_hidden_states.repeat_interleave(num_inference_steps, dim=0)
        img_attention_mask = torch.ones(
            (img_hidden_states.shape[0], img_hidden_states.shape[1]),
            device=latent_model_input.device,
            dtype=img_hidden_states.dtype
        )

        timesteps = ts.repeat(2).to(latent_model_input.device)
        res = model_base.sana(
            hidden_states=latent_model_input,
            encoder_hidden_states=img_hidden_states,
            timestep=timesteps,
            encoder_attention_mask=img_attention_mask,
            return_dict=False,
        )
        noise_pred = res[0]
        noise_pred_uncond, noise_pred= noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)
        return noise_pred
                # Save generated images
                # if dist.get_rank() == 0:  # Only save on rank 0 to avoid duplicates
                # save_dir = os.path.join(self.args.output_dir, "generated_images")
                # os.makedirs(save_dir, exist_ok=True)
                # rank = torch.distributed.get_rank()
                # step = self.state.global_step
                # for idx, (img, prompt) in enumerate(zip(images, prompts)):
                #     # Create filename: step_{step}_sample_{idx}.png
                #     filename = f"step_{step:06d}_sample_{idx}_rank_{rank}.png"
                #     save_path = os.path.join(save_dir, filename)
                #     img.save(save_path)

                # # Optionally save prompts to a text file
                # prompt_file = os.path.join(save_dir, f"step_{step:06d}_prompts.txt")
                # with open(prompt_file, "w") as f:
                #     for idx, prompt in enumerate(prompts):
                #         f.write(f"Sample {idx}: {prompt}\n")

        # return 0
            # _, images, traj_log_probs, diffusion_latents, traj_denoised_latents, traj_latents, ts = \
            #     self.old_model.prepare_prompts(
            #     vae_generation_input = self.old_model.prepare_vae_latent(
            #         curr_kvlens=,
            #         curr_rope=,
            #         image_sizes=,
            #         new_token_ids=
            #     )
            #     cfg_vae_generation_input = self.old_model.prepare_vae_latent_cfg(
            #         curr_kvlens=,
            #         curr_rope=,
            #         image_sizes=
            #     )
            #     self.old_model.generate_image(
            #         **vae_generation_input,
            #         num_timesteps=self.num_inference_steps,
            #     )
        # Compute rewards and advantages
        # rewards, rewards_per_func = self._compute_rewards(inputs, images)
        # reshaped_rewards = rewards.view(-1, self.num_generations)
        # mean_rewards = reshaped_rewards.mean(dim=1).repeat_interleave(self.num_generations)
        # if self.num_generations > 1:
        #     std_rewards = reshaped_rewards.std(dim=1).repeat_interleave(self.num_generations)
        # else:
        #     std_rewards = torch.zeros_like(mean_rewards)
        # advantages = (rewards - mean_rewards) / (std_rewards + 1e-4)
        # advantages = torch.clamp(advantages, -5, 5)
        
        # self._log_step(images, advantages, None)

        # policy_noise_preds = self._compute_diffusion_pred(
        #     model,
        #     diffusion_latents=diffusion_latents,
        #     traj_cur_latents=traj_latents,
        #     ts=ts,
        #     guidance_scale=self.guidance_scale,
        #     num_inference_steps=self.num_inference_steps,
        #     num_images_per_prompt=self.num_generations
        # )

        # with torch.no_grad():
        #     ref_noise_preds = self._compute_diffusion_pred(
        #         self.ref_model,
        #         diffusion_latents=diffusion_latents,
        #         traj_cur_latents=traj_latents,
        #         ts=ts,
        #         guidance_scale=self.guidance_scale,
        #         num_inference_steps=self.num_inference_steps,
        #         num_images_per_prompt=self.num_generations
        #     )
        
        # # Compute log probs and KL
        # _, policy_log_probs, policy_mean, policy_std = compute_log_prob(
        #     policy_noise_preds, model.get_scheduler(), traj_latents, traj_denoised_latents, ts
        # )
        # _, _, ref_mean, ref_std = compute_log_prob(
        #     ref_noise_preds, model.get_scheduler(), traj_latents, traj_denoised_latents, ts
        # )
        
        # kl = (policy_mean - ref_mean)**2 / (2 * policy_std**2)
        # kl = kl.mean(dim=tuple(range(1, kl.ndim)))
        
        # # GRPO loss
        # advantages_steps = advantages.repeat_interleave(
        #     self.num_inference_steps, dim=0
        # )
        # ratio = torch.exp(policy_log_probs - traj_log_probs)
        # unclipped_loss = -advantages_steps * ratio
        # clipped_loss = -advantages_steps * torch.clamp(ratio, 1.0 - 1e-4, 1.0 + 1e-4)
        # policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
        
        # loss = policy_loss + self.beta * kl.mean()
        
        # # Logging
        # self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
        # self._metrics["kl"].append(self.accelerator.gather_for_metrics(kl).mean().item())
        
        # for i, (func_name, _, _) in enumerate(self.reward_funcs):
        #     self._metrics[f"reward/{func_name}"].append(
        #         self.accelerator.gather_for_metrics(
        #             rewards_per_func[:, i]
        #         ).mean().item())
                
        # return loss


# @TRAINER_REGISTER.register("blip3o_next_t2icot_grpo_trainer")
# class T2ICoTGRPOTrainer(BaseBLIP3oGRPOTrainer):
#     """
#     GRPO Trainer for Text-to-Image with Chain-of-Thought reasoning.
    
#     This trainer combines CoT text generation with T2I diffusion, optimizing both
#     the reasoning process and the image generation jointly.
#     """
    
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
        
#         # Setup generation config for CoT
#         self.generation_config = {

#         }
    
#     # def _configure_parameters(self, model: PreTrainedModel):
#     #     """Configure parameters for CoT + T2I training."""
#     #     # Freeze reference model
#     #     for p in self.ref_model.parameters():
#     #         p.requires_grad = False
        
#     #     # Get model components
#     #     model_base = model.get_model()
        
#     #     # For CoT+T2I, we train both language and diffusion components
#     #     for p in model_base.parameters():
#     #         p.requires_grad = True
#     #     for p in model.visual.parameters():
#     #         p.requires_grad = True
#     #     for p in model.lm_head.parameters():
#     #         p.requires_grad = True
        
#     #     # T2I components
#     #     model_base.down_projector.requires_grad_(True)
#     #     model_base.t2i_queries.requires_grad = True
#     #     model_base.dit.requires_grad_(True)
    
#     # def create_optimizer(self):
#     #     """
#     #     Create optimizer with different learning rates for DiT and LLM.
#     #     DiT gets higher LR (1e-5), LLM gets lower LR (1e-6).
#     #     """
#     #     opt_kwargs = {
#     #         "betas": (self.args.adam_beta1, self.args.adam_beta2),
#     #         "eps": self.args.adam_epsilon,
#     #         "weight_decay": self.args.weight_decay,
#     #     }
        
#     #     dit_params = []
#     #     llm_params = []
        
#     #     # Collect DiT parameters
#     #     for name, param in self.model.get_model().dit.named_parameters():
#     #         if param.requires_grad:
#     #             dit_params.append(param)
        
#     #     for name, param in self.model.get_model().down_projector.named_parameters():
#     #         if param.requires_grad:
#     #             dit_params.append(param)
        
#     #     if self.model.get_model().t2i_queries.requires_grad:
#     #         dit_params.append(self.model.get_model().t2i_queries)
        
#     #     # Collect LLM parameters
#     #     for name, param in self.model.get_model().named_parameters():
#     #         if param.requires_grad and "dit" not in name and "queries" not in name and "down_projector" not in name:
#     #             llm_params.append(param)
        
#     #     # Create parameter groups with different LRs
#     #     param_groups = [
#     #         {"params": dit_params, "lr": 3e-6},
#     #         {"params": llm_params, "lr": 5e-7},
#     #     ]
        
#     #     optimizer = torch.optim.AdamW(param_groups, **opt_kwargs)
#     #     return optimizer
        
    
#     def _compute_cot_loss(
#         self,
#         model: PreTrainedModel,
#         prompt_completion_ids: torch.Tensor,
#         completion_ids: torch.Tensor,
#         advantages: torch.Tensor,
#         prompt_length: int
#     ):
#         """
#         Compute CoT loss on text generation.
        
#         Args:
#             model: Policy model
#             prompt_completion_ids: Full prompt+completion token IDs
#             completion_ids: Completion-only token IDs
#             advantages: Advantage values for each generation
#             prompt_length: Length of prompt tokens
            
#         Returns:
#             Tuple of (cot_loss, mean_kl, completion_mask)
#         """
#         def get_per_token_logps(model_class, input_ids):
#             """Get log probabilities for each token."""
#             logits = model_class(input_ids).logits[:, :-1, :]
#             input_ids = input_ids[:, 1:]
            
#             per_token_logps = []
#             for logits_row, input_ids_row in zip(logits, input_ids):
#                 log_probs = logits_row.log_softmax(dim=-1)
#                 token_log_prob = torch.gather(
#                     log_probs, dim=1, index=input_ids_row.unsqueeze(1)
#                 ).squeeze(1)
#                 per_token_logps.append(token_log_prob)
            
#             return torch.stack(per_token_logps)
        
#         # Policy log probs
#         per_token_logps = get_per_token_logps(model, prompt_completion_ids)
#         per_token_logps = per_token_logps[:, prompt_length - 1:]
        
#         # Reference log probs
#         with torch.inference_mode():
#             ref_per_token_logps = get_per_token_logps(self.ref_model, prompt_completion_ids)
#         ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1:]
        
#         # KL divergence
#         per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - \
#                       (ref_per_token_logps - per_token_logps) - 1
        
#         # Mask everything after first EOS token
#         is_eos = completion_ids == self.processing_class.eos_token_id
#         device = self.accelerator.device
#         eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
#         eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
#         sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
#         completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        
#         # Compute loss
#         advantages_cot = advantages.unsqueeze(1)
#         per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages_cot
#         per_token_loss = -(per_token_loss - 0.01 * per_token_kl)
#         cot_loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
#         mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        
#         return cot_loss, mean_kl, completion_mask
    
#     def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
#         if return_outputs:
#             raise ValueError("GRPOTrainer does not support returning outputs")
        
#         assert self.num_generations > 1, "num_generations must be greater than 1 for GRPO"
#         device = self.accelerator.device
#         tokenizer = self.train_dataset.processor.tokenizer
#         prompt_inputs = {
#             "input_ids": inputs["input_ids"][:, -self.max_prompt_length:],
#             "attention_mask": inputs["attention_mask"][:, -self.max_prompt_length:],
#         }

#         with torch.no_grad():
#             completion = self.old_model.generate_gen_ids(
#                 **prompt_inputs,
#                 max_new_tokens=None,
#                 do_sample=True,
#                 temperature=1.0,
#                 num_return_sequences=self.num_generations,
#                 pad_token_id=tokenizer.pad_token_id,
#                 eos_token_id=tokenizer.eos_token_id
#             )
#             # Pad if needed
#             max_length = completion.size(1)
#             prompt_completion_ids = completion
#             for completion in prompt_completion_ids:
#                 print(completion)
        
#         prompt_length = prompt_inputs["input_ids"].size(1)
#         completion_ids = prompt_completion_ids[:, prompt_length:]
#         # Decode completions for T2I
#         completions = tokenizer.batch_decode(
#             prompt_completion_ids, skip_special_tokens=True
#         )

#         with torch.no_grad():
#             _, images, traj_log_probs, diffusion_latents, traj_denoised_latents, traj_latents, ts = \
#                 self.old_model.generate_images(
#                     gen_ids=prompt_completion_ids,
#                     # attention_mask=inputs["attention_mask"],
#                     guidance_scale=self.guidance_scale,
#                     num_inference_steps=self.num_inference_steps,
#                     num_images_per_prompt=self.num_generations,
#                     use_sde=True
#                 )
#         # Compute rewards and advantages
#         rewards, rewards_per_func = self._compute_rewards(inputs, images, completions)
#         reshaped_rewards = rewards.view(-1, self.num_generations)
#         mean_rewards = reshaped_rewards.mean(dim=1).repeat_interleave(self.num_generations)
#         std_rewards = reshaped_rewards.std(dim=1).repeat_interleave(self.num_generations)
#         advantages = (rewards - mean_rewards) / (std_rewards + 1e-4)
#         advantages = torch.clamp(advantages, -5, 5)

#         # rewards, rewards_per_func = self._compute_rewards(inputs, images)
#         # reshaped_rewards = rewards.view(-1, self.num_generations)
#         # mean_rewards = reshaped_rewards.mean(dim=1).repeat_interleave(self.num_generations)
#         # std_rewards = reshaped_rewards.std(dim=1).repeat_interleave(self.num_generations)
#         # advantages = (rewards - mean_rewards) / (std_rewards + 1e-4)
#         # advantages = torch.clamp(advantages, -5, 5)
        
#         self._log_step(images, advantages, completions)

#         # CoT loss computation
#         cot_loss, mean_kl_cot, completion_mask = self._compute_cot_loss(
#             model, prompt_completion_ids, completion_ids, advantages, prompt_length
#         )

#         policy_noise_preds = self._compute_diffusion_pred(
#             model,
#             diffusion_latents=diffusion_latents,
#             traj_cur_latents=traj_latents,
#             ts=ts,
#             guidance_scale=self.guidance_scale,
#             num_inference_steps=self.num_inference_steps,
#             num_images_per_prompt=self.num_generations
#         )

#         with torch.no_grad():
#             ref_noise_preds = self._compute_diffusion_pred(
#                 self.ref_model,
#                 diffusion_latents=diffusion_latents,
#                 traj_cur_latents=traj_latents,
#                 ts=ts,
#                 guidance_scale=self.guidance_scale,
#                 num_inference_steps=self.num_inference_steps,
#                 num_images_per_prompt=self.num_generations
#             )
        
#         # Compute log probs and KL
#         _, policy_log_probs, policy_mean, policy_std = compute_log_prob(
#             policy_noise_preds, model.get_scheduler(), traj_latents, traj_denoised_latents, ts
#         )
#         _, _, ref_mean, ref_std = compute_log_prob(
#             ref_noise_preds, model.get_scheduler(), traj_latents, traj_denoised_latents, ts
#         )
        
#         kl = (policy_mean - ref_mean)**2 / (2 * policy_std**2)
#         kl = kl.mean(dim=tuple(range(1, kl.ndim)))
        
#         # GRPO loss
#         advantages_steps = advantages.repeat_interleave(self.num_inference_steps, dim=0)
#         ratio = torch.exp(policy_log_probs - traj_log_probs)
#         unclipped_loss_diff = -advantages_steps * ratio
#         clipped_loss_diff = -advantages_steps * torch.clamp(ratio, 1.0 - 1e-4, 1.0 + 1e-4)
#         policy_loss_diff = torch.mean(torch.maximum(unclipped_loss_diff, clipped_loss_diff))
#         diff_loss = policy_loss_diff + self.beta * kl.mean()
        
        
#         # Generate images from CoT completions
#         # with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
#         #     with torch.no_grad():
#         #         images, diff_log_probs_traj, prev_latents, pred_latents, ts = \
#         #             unwrapped_model.generate_image(
#         #                 text=completions,
#         #                 tokenizer=self.processing_class.tokenizer,
#         #                 diffusion_kwargs=self.diffusion_config,
#         #                 use_sde=True,
#         #             )
        
#         # # Compute rewards and advantages
#         # rewards, rewards_per_func = self._compute_rewards(inputs, images, completions)
#         # reshaped_rewards = rewards.view(-1, self.num_generations)
#         # mean_rewards = reshaped_rewards.mean(dim=1).repeat_interleave(self.num_generations)
#         # std_rewards = reshaped_rewards.std(dim=1).repeat_interleave(self.num_generations)
#         # advantages = (rewards - mean_rewards) / (std_rewards + 1e-4)
#         # advantages = torch.clamp(advantages, -5, 5)
        
        
#         # self._log_step(images, prompts_text, advantages, completions)

#         # # CoT loss computation
#         # cot_loss, mean_kl_cot, completion_mask = self._compute_cot_loss(
#         #     model, prompt_completion_ids, completion_ids, advantages, prompt_length
#         # )
        
#         # # Diffusion loss computation
#         # model_pred = self._compute_diffusion_loss(
#         #     model, completions, prev_latents, pred_latents, ts, "t2i_queries", 1
#         # )
        
#         # with torch.no_grad():
#         #     ref_model_pred = self._compute_diffusion_loss(
#         #         self.ref_model, completions, prev_latents, pred_latents, ts, "t2i_queries", 1
#         #     )
        
#         # # Compute log probs and KL for diffusion
#         # _, log_prob_diff, mean_diff, std_diff = compute_log_prob(
#         #     model_pred, self.scheduler, prev_latents, pred_latents, ts
#         # )
#         # _, _, mean_ref_diff, std_ref_diff = compute_log_prob(
#         #     ref_model_pred, self.scheduler, prev_latents, pred_latents, ts
#         # )
        
#         # kl_diff = (mean_diff - mean_ref_diff)**2 / (2 * std_diff**2)
#         # kl_diff = kl_diff.mean(dim=tuple(range(1, kl_diff.ndim)))
        
#         # # Diffusion GRPO loss
#         # advantages_diff = advantages.repeat_interleave(
#         #     self.diffusion_config["num_inference_steps"], dim=0
#         # )
#         # ratio_diff = torch.exp(log_prob_diff - diff_log_probs_traj)
#         # assert (ratio_diff == 1).all(), f"{ratio_diff}"

#         # unclipped_loss_diff = -advantages_diff * ratio_diff
#         # clipped_loss_diff = -advantages_diff * torch.clamp(
#         #     ratio_diff, 1.0 - 1e-4, 1.0 + 1e-4
#         # )
#         # diff_loss = torch.mean(torch.maximum(unclipped_loss_diff, clipped_loss_diff))
#         # diff_loss = diff_loss + self.beta * kl_diff.mean()
        
#         # Combined loss
#         loss = cot_loss + diff_loss
        
#         # Logging
#         completion_length = self.accelerator.gather_for_metrics(
#             completion_mask.sum(1)
#         ).float().mean().item()
        
#         self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
#         self._metrics["cot_loss"].append(self.accelerator.gather_for_metrics(cot_loss).mean().item())
#         self._metrics["cot_kl"].append(self.accelerator.gather_for_metrics(mean_kl_cot).mean().item())
#         self._metrics["diff_loss"].append(self.accelerator.gather_for_metrics(diff_loss).mean().item())
#         self._metrics["diff_kl"].append(self.accelerator.gather_for_metrics(kl_diff).mean().item())
#         self._metrics["completion_length"].append(completion_length)
        
#         for i, (func_name, _, _) in enumerate(self.reward_funcs):
#             self._metrics[f"reward/{func_name}"].append(
#                 self.accelerator.gather_for_metrics(
#                     rewards_per_func[:, i]
#                 ).mean().item()
#             )
        
#         return loss