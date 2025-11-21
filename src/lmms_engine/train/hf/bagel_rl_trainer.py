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
                unpacked_latent, traj_log_probs, traj_denoised_x_t, traj_cur_x_t, ts = unwrapped_model.generate_image(
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

        policy_traj_v_t = self._compute_diffusion_pred(
            model,
            traj_x_t=traj_cur_x_t,
            timesteps=ts,
            packed_text_ids=generation_input["packed_text_ids"],
            packed_text_indexes=generation_input["packed_text_indexes"],
            packed_vae_position_ids=generation_input["packed_vae_position_ids"],
            packed_vae_token_indexes=generation_input["packed_vae_token_indexes"],
            packed_seqlens=generation_input["packed_seqlens"],
            packed_position_ids=generation_input["packed_position_ids"],
            packed_indexes=generation_input["packed_indexes"],
            past_key_values=past_key_values,
            key_values_lens=generation_input["key_values_lens"],
            packed_key_value_indexes=generation_input["packed_key_value_indexes"],
            cfg_text_scale=cfg_text_scale,
            cfg_text_packed_query_indexes=cfg_generation_input["cfg_packed_query_indexes"],
            cfg_text_packed_position_ids=cfg_generation_input["cfg_packed_position_ids"],
            cfg_text_past_key_values=cfg_text_past_key_values,
            cfg_text_key_values_lens=cfg_generation_input["cfg_key_values_lens"],
            cfg_text_packed_key_value_indexes=cfg_generation_input["cfg_packed_key_value_indexes"],
        )

        if self.ref_model is not None:
            with torch.no_grad():
                ref_traj_v_t = self._compute_diffusion_pred(
                    self.ref_model,
                    traj_x_t=traj_cur_x_t,
                    timesteps=ts,
                    packed_text_ids=generation_input["packed_text_ids"],
                    packed_text_indexes=generation_input["packed_text_indexes"],
                    packed_vae_position_ids=generation_input["packed_vae_position_ids"],
                    packed_vae_token_indexes=generation_input["packed_vae_token_indexes"],
                    packed_seqlens=generation_input["packed_seqlens"],
                    packed_position_ids=generation_input["packed_position_ids"],
                    packed_indexes=generation_input["packed_indexes"],
                    past_key_values=past_key_values,
                    key_values_lens=generation_input["key_values_lens"],
                    packed_key_value_indexes=generation_input["packed_key_value_indexes"],
                    cfg_text_scale=cfg_text_scale,
                    cfg_text_packed_query_indexes=cfg_generation_input["cfg_packed_query_indexes"],
                    cfg_text_packed_position_ids=cfg_generation_input["cfg_packed_position_ids"],
                    cfg_text_past_key_values=cfg_text_past_key_values,
                    cfg_text_key_values_lens=cfg_generation_input["cfg_key_values_lens"],
                    cfg_text_packed_key_value_indexes=cfg_generation_input["cfg_packed_key_value_indexes"],
                )
        
        # Compute log probs and KL
        _, policy_log_probs, policy_mean, policy_std = compute_log_prob(
            policy_traj_v_t, self.scheduler, traj_cur_x_t, traj_denoised_x_t, ts
        )
        if self.ref_model is not None:
            _, _, ref_mean, ref_std = compute_log_prob(
                ref_traj_v_t, self.scheduler, traj_cur_x_t, traj_denoised_x_t, ts
            )
        
            kl = (policy_mean - ref_mean)**2 / (2 * policy_std**2)
            kl = kl.mean(dim=tuple(range(1, kl.ndim)))
            self._metrics["kl"].append(self.accelerator.gather_for_metrics(kl).mean().item())
        
        # GRPO loss
        advantages_steps = advantages.repeat_interleave(
            self.num_inference_steps, dim=0
        )
        ratio = torch.exp(policy_log_probs - traj_log_probs)
        unclipped_loss = -advantages_steps * ratio
        clipped_loss = -advantages_steps * torch.clamp(ratio, 1.0 - 1e-4, 1.0 + 1e-4)
        policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
        
        loss = policy_loss + self.beta * kl.mean() if self.ref_model is not None else policy_loss
        
        # Logging
        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
        
        for i, (func_name, _, _) in enumerate(self.reward_funcs):
            self._metrics[f"reward/{func_name}"].append(
                self.accelerator.gather_for_metrics(
                    rewards_per_func[:, i]
                ).mean().item())
                
        return loss
    
    def _compute_diffusion_pred(
        self, 
        model_to_use: PreTrainedModel, 
        traj_x_t: torch.Tensor,
        timesteps: torch.Tensor,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_vae_position_ids: torch.LongTensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_position_ids: torch.LongTensor,
        packed_indexes: torch.LongTensor,
        past_key_values: NaiveCache,
        key_values_lens: torch.IntTensor,
        packed_key_value_indexes: torch.LongTensor,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        cfg_interval: Optional[Tuple[float, float]] = (0.0, 1.0),
        # cfg_text
        cfg_text_scale: float = 1.0,
        cfg_text_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_text_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_text_past_key_values: Optional[NaiveCache] = None,
        cfg_text_key_values_lens: Optional[torch.IntTensor] = None,
        cfg_text_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        # cfg_img
        cfg_img_scale: float = 1.0,
        cfg_img_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_img_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_img_past_key_values: Optional[NaiveCache] = None,
        cfg_img_key_values_lens: Optional[torch.IntTensor] = None,
        cfg_img_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        cfg_type: str = "parallel",
        # sde_scheduler=None,
    ):
        # if sde_scheduler is None:
        #     sde_scheduler = self.scheduler
        #     if isinstance(sde_scheduler, FlowMatchEulerDiscreteScheduler):
        #         sigmas = np.linspace(1.0, 1 / num_timesteps, num_timesteps)
        #         sde_scheduler.set_timesteps(num_timesteps, sigmas=sigmas)
        #     else:
        #         sde_scheduler.set_timesteps(num_timesteps)

        traj_v_t = []

        for i, t in enumerate(timesteps):
            x_t = traj_x_t[i]
            timestep = torch.tensor([t] * x_t.shape[0], device=x_t.device)
            if t > cfg_interval[0] and t <= cfg_interval[1]:
                cfg_text_scale_ = cfg_text_scale
                cfg_img_scale_ = cfg_img_scale
            else:
                cfg_text_scale_ = 1.0
                cfg_img_scale_ = 1.0

            v_t = self._forward_flow(
                model=model_to_use,
                x_t=x_t,
                timestep=timestep,
                packed_vae_token_indexes=packed_vae_token_indexes,
                packed_vae_position_ids=packed_vae_position_ids,
                packed_text_ids=packed_text_ids,
                packed_text_indexes=packed_text_indexes,
                packed_position_ids=packed_position_ids,
                packed_indexes=packed_indexes,
                packed_seqlens=packed_seqlens,
                key_values_lens=key_values_lens,
                past_key_values=past_key_values,
                packed_key_value_indexes=packed_key_value_indexes,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                # cfg_text
                cfg_text_scale=cfg_text_scale_,
                cfg_text_packed_position_ids=cfg_text_packed_position_ids,
                cfg_text_packed_query_indexes=cfg_text_packed_query_indexes,
                cfg_text_key_values_lens=cfg_text_key_values_lens,
                cfg_text_past_key_values=cfg_text_past_key_values,
                cfg_text_packed_key_value_indexes=cfg_text_packed_key_value_indexes,
                # cfg_img
                cfg_img_scale=cfg_img_scale_,
                cfg_img_packed_position_ids=cfg_img_packed_position_ids,
                cfg_img_packed_query_indexes=cfg_img_packed_query_indexes,
                cfg_img_key_values_lens=cfg_img_key_values_lens,
                cfg_img_past_key_values=cfg_img_past_key_values,
                cfg_img_packed_key_value_indexes=cfg_img_packed_key_value_indexes,
                cfg_type=cfg_type,
            )
            traj_v_t.append(v_t.unsqueeze(0))

        traj_v_t = torch.cat(traj_v_t, dim=0)
        return traj_v_t

    def _forward_flow(
        self,
        model: PreTrainedModel,
        x_t: torch.Tensor,
        timestep: torch.LongTensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_vae_position_ids: torch.LongTensor,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_indexes: torch.LongTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        key_values_lens: torch.IntTensor,
        past_key_values: NaiveCache,
        packed_key_value_indexes: torch.LongTensor,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        # cfg_text
        cfg_text_scale: float = 1.0,
        cfg_text_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_text_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_text_key_values_lens: Optional[torch.Tensor] = None,
        cfg_text_past_key_values: Optional[NaiveCache] = None,
        cfg_text_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        # cfg_img
        cfg_img_scale: float = 1.0,
        cfg_img_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_img_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_img_key_values_lens: Optional[torch.Tensor] = None,
        cfg_img_past_key_values: Optional[NaiveCache] = None,
        cfg_img_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        cfg_type: str = "parallel",
    ):
        packed_text_embedding = model.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), model.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        assert timestep.unique().shape[0] == 1
        packed_pos_embed = model.latent_pos_embed(packed_vae_position_ids)
        packed_timestep_embeds = model.time_embedder(timestep)
        x_t = model.vae2llm(x_t) + packed_timestep_embeds + packed_pos_embed
        if x_t.dtype != packed_sequence.dtype:
            x_t = x_t.to(packed_sequence.dtype)
        packed_sequence[packed_vae_token_indexes] = x_t

        extra_inputs = {}
        if model.use_moe:
            extra_inputs = {
                "mode": "gen",
                "packed_vae_token_indexes": packed_vae_token_indexes,
                "packed_text_indexes": packed_text_indexes,
            }

        output = model.language_model.forward_inference(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=False,
            is_causal=False,
            **extra_inputs,
        )
        v_t = model.llm2vae(output.packed_query_sequence)
        v_t = v_t[packed_vae_token_indexes]

        if cfg_text_scale > 1.0:
            cfg_text_output = model.language_model.forward_inference(
                packed_query_sequence=packed_sequence,
                query_lens=packed_seqlens,
                packed_query_position_ids=cfg_text_packed_position_ids,
                packed_query_indexes=cfg_text_packed_query_indexes,
                past_key_values=cfg_text_past_key_values,
                key_values_lens=cfg_text_key_values_lens,
                packed_key_value_indexes=cfg_text_packed_key_value_indexes,
                update_past_key_values=False,
                is_causal=False,
                **extra_inputs,
            )
            cfg_text_v_t = model.llm2vae(cfg_text_output.packed_query_sequence)
            cfg_text_v_t = cfg_text_v_t[packed_vae_token_indexes]

        if cfg_img_scale > 1.0:
            cfg_img_output = model.language_model.forward_inference(
                packed_query_sequence=packed_sequence,
                query_lens=packed_seqlens,
                packed_query_position_ids=cfg_img_packed_position_ids,
                packed_query_indexes=cfg_img_packed_query_indexes,
                past_key_values=cfg_img_past_key_values,
                key_values_lens=cfg_img_key_values_lens,
                packed_key_value_indexes=cfg_img_packed_key_value_indexes,
                update_past_key_values=False,
                is_causal=False,
                **extra_inputs,
            )
            cfg_img_v_t = model.llm2vae(cfg_img_output.packed_query_sequence)
            cfg_img_v_t = cfg_img_v_t[packed_vae_token_indexes]

        if cfg_text_scale > 1.0:
            if cfg_renorm_type == "text_channel":
                v_t_text_ = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)
                norm_v_t = torch.norm(v_t, dim=-1, keepdim=True)
                norm_v_t_text_ = torch.norm(v_t_text_, dim=-1, keepdim=True)
                scale = (norm_v_t / (norm_v_t_text_ + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
                v_t_text = v_t_text_ * scale
                if cfg_img_scale > 1.0:
                    v_t = cfg_img_v_t + cfg_img_scale * (v_t_text - cfg_img_v_t)
                else:
                    v_t = v_t_text
            else:
                v_t_text_ = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)

                if cfg_img_scale > 1.0:
                    v_t_ = cfg_img_v_t + cfg_img_scale * (v_t_text_ - cfg_img_v_t)
                else:
                    v_t_ = v_t_text_

                # NOTE norm is computed over all dimensions, thus currently only supports batch_size = 1 with navit
                if cfg_renorm_type == "global":
                    norm_v_t = torch.norm(v_t)
                    norm_v_t_ = torch.norm(v_t_)
                elif cfg_renorm_type == "channel":
                    norm_v_t = torch.norm(v_t, dim=-1, keepdim=True)
                    norm_v_t_ = torch.norm(v_t_, dim=-1, keepdim=True)
                else:
                    raise NotImplementedError(f"{cfg_renorm_type} is not supported")
                scale = (norm_v_t / (norm_v_t_ + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
                v_t = v_t_ * scale
        else:
            # No CFG
            pass

        return v_t