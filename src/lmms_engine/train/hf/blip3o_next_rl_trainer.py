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

from contextlib import contextmanager
from lmms_engine.models.blip3o_next import sde_step_with_logprob, blip3oQwenForInferenceLM
from lmms_engine.train.registry import TRAINER_REGISTER
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

if is_wandb_available():
    import wandb

UND_IMAGE_TOKEN = "<|image_pad|>"
UND_IMAGE_TOKEN_IDX = 151655

# LAYER_PATTERNS = [
#     "transformer.h.{layer}",
#     "model.decoder.layers.{layer}",
#     "gpt_neox.layers.{layer}",
#     "model.layers.{layer}",
# ]

# def maybe_apply_chat_template(
#     example: dict[str, list[dict[str, str]]],
#     tokenizer: PreTrainedTokenizerBase,
#     tools: list[dict | Callable] | None = None,
#     **template_kwargs: Any,
# ) -> dict[str, str]:
#     if is_conversational(example):
#         return apply_chat_template(example, tokenizer, tools, **template_kwargs)
#     else:
#         return example

# def prepare_deepspeed(
#     model: torch.nn.Module, per_device_train_batch_size: int, fp16: bool = False, bf16: bool = False
# ) -> torch.nn.Module:
    # import deepspeed
    # deepspeed_plugin = AcceleratorState().deepspeed_plugin
    # config_kwargs = deepspeed_plugin.deepspeed_config
    # if config_kwargs["zero_optimization"]["stage"] != 3:
    #     config_kwargs["train_micro_batch_size_per_gpu"] = per_device_train_batch_size
    #     config_kwargs = {
    #         "train_micro_batch_size_per_gpu": config_kwargs["train_micro_batch_size_per_gpu"],
    #         "prescale_gradients": False,
    #         "wall_clock_breakdown": False,
    #     }
    #     if bf16:
    #         config_kwargs["bf16"] = {"enabled": True}
    #     elif fp16:
    #         config_kwargs["fp16"] = {"enabled": True}
    # else:
    #     if hasattr(model, "config"):
    #         hidden_size = (
    #             max(model.config.hidden_sizes)
    #             if getattr(model.config, "hidden_sizes", None)
    #             else getattr(model.config, "hidden_size", None)
    #         )
    #         if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
    #             # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
    #             # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
    #             config_kwargs.update(
    #                 {
    #                     "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
    #                     "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
    #                     "zero_optimization.stage3_prefetch_bucket_size": 0,
    #                 }
    #             )
    # model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
    # model.eval()
    # return model

# @contextmanager
# def unwrap_model_for_generation(
#     model: "DistributedDataParallel | DeepSpeedEngine",
#     accelerator: "Accelerator",
#     gather_deepspeed3_params: bool = True,
# ):
#     unwrapped_model = accelerator.unwrap_model(model)
#     is_gradient_checkpointing = unwrapped_model.is_gradient_checkpointing
#     if is_gradient_checkpointing:
#         unwrapped_model.gradient_checkpointing_disable()
#     if accelerator.state.deepspeed_plugin is not None and accelerator.state.deepspeed_plugin.zero_stage == 3:
#         if not gather_deepspeed3_params:
#             yield accelerator.unwrap_model(model)
#         else:
#             import deepspeed
#             with deepspeed.zero.GatheredParameters(model.parameters()):
#                 remove_hooks(model)
#                 yield accelerator.unwrap_model(model)
#                 add_hooks(model)
#     else:
#         yield unwrapped_model
#     if is_gradient_checkpointing:
#         unwrapped_model.gradient_checkpointing_enable()

# def create_reference_model(
#     model: PreTrainedModelWrapper, num_shared_layers: int | None = None, pattern: str | None = None
# ) -> PreTrainedModelWrapper:
    # """
    # Creates a static reference copy of a model. Note that model will be in `.eval()` mode.

    # Args:
    #     model ([`PreTrainedModelWrapper`]): The model to be copied.
    #     num_shared_layers (`int`, *optional*):
    #         The number of initial layers that are shared between both models and kept frozen.
    #     pattern (`str`, *optional*): The shared layers are selected with a string pattern
    #         (e.g. "transformer.h.{layer}" for GPT2) and if a custom pattern is necessary it can be passed here.

    # Returns:
    #     [`PreTrainedModelWrapper`]
    # """
    # if is_deepspeed_zero3_enabled():
    #     raise ValueError(
    #         "DeepSpeed ZeRO-3 is enabled and is not compatible with `create_reference_model()`. Please instantiate your reference model directly with `AutoModelForCausalLM.from_pretrained()`."
    #     )

    # parameter_names = [n for n, _ in model.named_parameters()]
    # ref_model = deepcopy(model)

    # # if no layers are shared, return copy of model
    # if num_shared_layers is None:
    #     for param_name in parameter_names:
    #         param = ref_model.get_parameter(param_name)
    #         param.requires_grad = False
    #     return ref_model.eval()

    # # identify layer name pattern
    # if pattern is not None:
    #     pattern = pattern.format(layer=num_shared_layers)
    # else:
    #     for pattern_candidate in LAYER_PATTERNS:
    #         pattern_candidate = pattern_candidate.format(layer=num_shared_layers)
    #         if any(pattern_candidate in name for name in parameter_names):
    #             pattern = pattern_candidate
    #             break

    # if pattern is None:
    #     raise ValueError("Layer pattern could not be matched.")

    # # divide parameters in shared and unshared parameter lists
    # shared_param_list = []
    # unshared_param_list = []

    # shared_parameter = True
    # for name, _param in model.named_parameters():
    #     if pattern in name:
    #         shared_parameter = False
    #     if shared_parameter:
    #         shared_param_list.append(name)
    #     else:
    #         unshared_param_list.append(name)

    # # create reference of the original parameter if they are shared
    # for param_name in shared_param_list:
    #     param = model.get_parameter(param_name)
    #     param.requires_grad = False

    #     _ref_param = ref_model.get_parameter(param_name)

    # # for all other parameters just make sure they don't use gradients
    # for param_name in unshared_param_list:
    #     param = ref_model.get_parameter(param_name)
    #     param.requires_grad = False

    # if pattern is not None and len(unshared_param_list) == 0:
    #     logging.warning("Pattern passed or found, but no layers matched in the model. Check for a typo.")

    # return ref_model.eval()

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

class BaseBLIP3oGRPOTrainer(Trainer):
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
        self.guidance_scale = rl_config.get("guidance_scale", 2)
        self.num_inference_steps = rl_config.get("num_inference_steps", 30)
        self.beta = float(rl_config.get("beta", 0.0))
        self.max_prompt_length = rl_config.get("max_prompt_length", 1024) # for t2i task, it will not be used.
        self.max_completion_length = rl_config.get("max_completion_length", 2048) # for t2i task, it will not be used.

        # Scheduler
        # self.scheduler = FlowMatchEulerDiscreteScheduler(shift=3.0)
        # self.scheduler.set_timesteps(self.num_inference_steps)

        # Image transforms
        self._setup_transforms()
        
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

        # Reference model
        model = kwargs.get("model", None)
        if is_deepspeed_zero3_enabled():
            assert model, "model is required for GRPO Trainer"
            self.ref_model = deepcopy(model)
            self.old_model = deepcopy(model)
        else:
            self.ref_model = create_reference_model(model)
            self.old_model = create_reference_model(model)
        # Freeze/unfreeze parameters based on task
        self._configure_parameters(model)

        super().__init__(*args, **kwargs)
        # Prepare reference model
        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
                self.old_model = prepare_deepspeed(self.old_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
                self.old_model = self.accelerator.prepare_model(self.old_model, evaluation_mode=True)
    
    def _configure_parameters(self, model: PreTrainedModel):
        """Configure which parameters to train based on task type."""
        # Freeze reference model
        for p in self.ref_model.parameters():
            p.requires_grad = False
        for p in self.old_model.parameters():
            p.requires_grad = False
        
        # Get model components
        model_base = model.get_model()
        
        # Freeze base components
        for p in model_base.parameters():
            p.requires_grad = False
        for p in model_base.vision_tower.parameters():
            p.requires_grad = False
        for p in model.lm_head.parameters():
            p.requires_grad = False
        
        # Unfreeze task-specific components
        if self.task_type == "t2i":
            model_base.diffusion_connector.requires_grad_(True)
            # model_base.t2i_queries.requires_grad = True
            model_base.sana.requires_grad_(True)
            del model_base.vision_tower
            del model_base.sana_vae.encoder
            del self.ref_model.get_model().vision_tower
            del self.ref_model.get_model().sana_vae.encoder
            del self.old_model.get_model().vision_tower
            del self.old_model.get_model().sana_vae.encoder
        elif self.task_type == "i2i":
            model_base.diffusion_connector.requires_grad_(True)
            # model_base.i2i_queries.requires_grad = True
            model_base.sana.requires_grad_(True)
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")

    def _setup_transforms(self):
        """Setup image transformation pipelines."""
        self.ref_trsf = T.Compose([
            T.Lambda(lambda x: x.convert("RGB")),
            T.Resize(1024, interpolation=InterpolationMode.BICUBIC, antialias=True),
            T.CenterCrop(1024),
            T.ToTensor(),
            T.Lambda(lambda x: x * 2 - 1)
        ])
        
        self.und_trsf = T.Compose([
            T.Lambda(lambda x: x.convert("RGB")),
            T.Resize(672, interpolation=InterpolationMode.BICUBIC, antialias=True),
            T.CenterCrop(672)
        ])
    
    def _set_signature_columns_if_needed(self):
        """Set required dataset columns."""
        if self._signature_columns is None:
            self._signature_columns = ["input_ids", "caption"]
    
    def _prepare_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Skip automatic tensor conversion."""
        return inputs


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

@TRAINER_REGISTER.register("blip3o_next_t2i_grpo_trainer")
class T2IGRPOTrainer(BaseBLIP3oGRPOTrainer):
    """GRPO Trainer for Text-to-Image generation."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute T2I GRPO loss."""
        if return_outputs:
            raise ValueError("GRPOTrainer does not support returning outputs")
        
        device = self.accelerator.device
        
        if self.state.global_step % 10 == 0 and self.state.global_step > 0:
            self._update_old_model(model)


        with torch.no_grad():
            _, images, traj_log_probs, diffusion_latents, traj_denoised_latents, traj_latents, ts = \
                self.old_model.generate_images(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    guidance_scale=self.guidance_scale,
                    num_inference_steps=self.num_inference_steps,
                    num_images_per_prompt=self.num_generations,
                    use_sde=True
                )
        # Compute rewards and advantages
        rewards, rewards_per_func = self._compute_rewards(inputs, images)
        reshaped_rewards = rewards.view(-1, self.num_generations)
        mean_rewards = reshaped_rewards.mean(dim=1).repeat_interleave(self.num_generations)
        if self.num_generations > 1:
            std_rewards = reshaped_rewards.std(dim=1).repeat_interleave(self.num_generations)
        else:
            std_rewards = torch.zeros_like(mean_rewards)
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
        diffusion_latents: torch.Tensor,
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


@TRAINER_REGISTER.register("blip3o_next_t2icot_grpo_trainer")
class T2ICoTGRPOTrainer(BaseBLIP3oGRPOTrainer):
    """
    GRPO Trainer for Text-to-Image with Chain-of-Thought reasoning.
    
    This trainer combines CoT text generation with T2I diffusion, optimizing both
    the reasoning process and the image generation jointly.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Setup generation config for CoT
        self.generation_config = {
            "max_new_tokens": 512,
            "do_sample": True,
            "temperature": 1.0,
            "num_return_sequences": self.num_generations,
            "pad_token_id": self.processing_class.pad_token_id,
            "eos_token_id": self.processing_class.eos_token_id,
        }
    
    def _configure_parameters(self, model: PreTrainedModel):
        """Configure parameters for CoT + T2I training."""
        # Freeze reference model
        for p in self.ref_model.parameters():
            p.requires_grad = False
        
        # Get model components
        model_base = model.get_model()
        
        # For CoT+T2I, we train both language and diffusion components
        for p in model_base.parameters():
            p.requires_grad = True
        for p in model.visual.parameters():
            p.requires_grad = True
        for p in model.lm_head.parameters():
            p.requires_grad = True
        
        # T2I components
        model_base.down_projector.requires_grad_(True)
        model_base.t2i_queries.requires_grad = True
        model_base.dit.requires_grad_(True)
    
    def create_optimizer(self):
        """
        Create optimizer with different learning rates for DiT and LLM.
        DiT gets higher LR (1e-5), LLM gets lower LR (1e-6).
        """
        opt_kwargs = {
            "betas": (self.args.adam_beta1, self.args.adam_beta2),
            "eps": self.args.adam_epsilon,
            "weight_decay": self.args.weight_decay,
        }
        
        dit_params = []
        llm_params = []
        
        # Collect DiT parameters
        for name, param in self.model.get_model().dit.named_parameters():
            if param.requires_grad:
                dit_params.append(param)
        
        for name, param in self.model.get_model().down_projector.named_parameters():
            if param.requires_grad:
                dit_params.append(param)
        
        if self.model.get_model().t2i_queries.requires_grad:
            dit_params.append(self.model.get_model().t2i_queries)
        
        # Collect LLM parameters
        for name, param in self.model.get_model().named_parameters():
            if param.requires_grad and "dit" not in name and "queries" not in name and "down_projector" not in name:
                llm_params.append(param)
        
        # Create parameter groups with different LRs
        param_groups = [
            {"params": dit_params, "lr": 3e-6},
            {"params": llm_params, "lr": 5e-7},
        ]
        
        optimizer = torch.optim.AdamW(param_groups, **opt_kwargs)
        return optimizer
        
    
    def _compute_cot_loss(
        self,
        model: PreTrainedModel,
        prompt_completion_ids: torch.Tensor,
        completion_ids: torch.Tensor,
        advantages: torch.Tensor,
        prompt_length: int
    ):
        """
        Compute CoT loss on text generation.
        
        Args:
            model: Policy model
            prompt_completion_ids: Full prompt+completion token IDs
            completion_ids: Completion-only token IDs
            advantages: Advantage values for each generation
            prompt_length: Length of prompt tokens
            
        Returns:
            Tuple of (cot_loss, mean_kl, completion_mask)
        """
        def get_per_token_logps(model_class, input_ids):
            """Get log probabilities for each token."""
            logits = model_class(input_ids).logits[:, :-1, :]
            input_ids = input_ids[:, 1:]
            
            per_token_logps = []
            for logits_row, input_ids_row in zip(logits, input_ids):
                log_probs = logits_row.log_softmax(dim=-1)
                token_log_prob = torch.gather(
                    log_probs, dim=1, index=input_ids_row.unsqueeze(1)
                ).squeeze(1)
                per_token_logps.append(token_log_prob)
            
            return torch.stack(per_token_logps)
        
        # Policy log probs
        per_token_logps = get_per_token_logps(model, prompt_completion_ids)
        per_token_logps = per_token_logps[:, prompt_length - 1:]
        
        # Reference log probs
        with torch.inference_mode():
            ref_per_token_logps = get_per_token_logps(self.ref_model, prompt_completion_ids)
        ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1:]
        
        # KL divergence
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - \
                      (ref_per_token_logps - per_token_logps) - 1
        
        # Mask everything after first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        device = self.accelerator.device
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        
        # Compute loss
        advantages_cot = advantages.unsqueeze(1)
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages_cot
        per_token_loss = -(per_token_loss - 0.01 * per_token_kl)
        cot_loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        
        return cot_loss, mean_kl, completion_mask
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute CoT + T2I GRPO loss."""
        if return_outputs:
            raise ValueError("GRPOTrainer does not support returning outputs")
        
        assert self.num_generations > 1, "num_generations must be greater than 1 for GRPO"
        device = self.accelerator.device
        # Prepare prompts for CoT generation
        # prompts_text = [
        #     maybe_apply_chat_template(ex, self.processing_class)["prompt"] 
        #     for ex in inputs
        # ]
        
        # prompt_inputs = self.processing_class.tokenizer(
        #     prompts_text,
        #     return_tensors="pt",
        #     padding=True,
        #     padding_side="left",
        #     add_special_tokens=False,
        # ).to(device)
        
        # if self.max_prompt_length is not None:
        #     prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -self.max_prompt_length:]
        #     prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -self.max_prompt_length:]
        
        # Generate CoT completions
        # with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
        with torch.no_grad():
            from transformers import GenerationConfig
            gen_config = GenerationConfig(**self.generation_config)
            completion = unwrapped_model.generate(**prompt_inputs, generation_config=gen_config)
            
            # Pad if needed
            max_length = completion.size(1)
            prompt_completion_ids = completion
        
        # prompt_length = prompt_inputs["input_ids"].size(1)
        # completion_ids = prompt_completion_ids[:, prompt_length:]
        
        # # Decode completions for T2I
        # completions = self.processing_class.tokenizer.batch_decode(
        #     prompt_completion_ids, skip_special_tokens=True
        # )

        with torch.no_grad():
            _, images, traj_log_probs, diffusion_latents, traj_denoised_latents, traj_latents, ts = \
                self.old_model.generate_images(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    guidance_scale=self.guidance_scale,
                    num_inference_steps=self.num_inference_steps,
                    num_images_per_prompt=self.num_generations,
                    use_sde=True
                )
        # Compute rewards and advantages
        rewards, rewards_per_func = self._compute_rewards(inputs, images, completions)
        reshaped_rewards = rewards.view(-1, self.num_generations)
        mean_rewards = reshaped_rewards.mean(dim=1).repeat_interleave(self.num_generations)
        std_rewards = reshaped_rewards.std(dim=1).repeat_interleave(self.num_generations)
        advantages = (rewards - mean_rewards) / (std_rewards + 1e-4)
        advantages = torch.clamp(advantages, -5, 5)

        # rewards, rewards_per_func = self._compute_rewards(inputs, images)
        # reshaped_rewards = rewards.view(-1, self.num_generations)
        # mean_rewards = reshaped_rewards.mean(dim=1).repeat_interleave(self.num_generations)
        # std_rewards = reshaped_rewards.std(dim=1).repeat_interleave(self.num_generations)
        # advantages = (rewards - mean_rewards) / (std_rewards + 1e-4)
        # advantages = torch.clamp(advantages, -5, 5)
        
        self._log_step(images, advantages, completions)


        # CoT loss computation
        cot_loss, mean_kl_cot, completion_mask = self._compute_cot_loss(
            model, prompt_completion_ids, completion_ids, advantages, prompt_length
        )

        policy_noise_preds = self._compute_diffusion_pred(
            model,
            diffusion_latents=diffusion_latents,
            traj_cur_latents=traj_latents,
            ts=ts,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
            num_images_per_prompt=self.num_generations
        )

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
        advantages_steps = advantages.repeat_interleave(self.num_inference_steps, dim=0)
        ratio = torch.exp(policy_log_probs - traj_log_probs)
        unclipped_loss_diff = -advantages_steps * ratio
        clipped_loss_diff = -advantages_steps * torch.clamp(ratio, 1.0 - 1e-4, 1.0 + 1e-4)
        policy_loss_diff = torch.mean(torch.maximum(unclipped_loss_diff, clipped_loss_diff))
        diff_loss = policy_loss_diff + self.beta * kl.mean()
        
        
        # Generate images from CoT completions
        # with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
        #     with torch.no_grad():
        #         images, diff_log_probs_traj, prev_latents, pred_latents, ts = \
        #             unwrapped_model.generate_image(
        #                 text=completions,
        #                 tokenizer=self.processing_class.tokenizer,
        #                 diffusion_kwargs=self.diffusion_config,
        #                 use_sde=True,
        #             )
        
        # # Compute rewards and advantages
        # rewards, rewards_per_func = self._compute_rewards(inputs, images, completions)
        # reshaped_rewards = rewards.view(-1, self.num_generations)
        # mean_rewards = reshaped_rewards.mean(dim=1).repeat_interleave(self.num_generations)
        # std_rewards = reshaped_rewards.std(dim=1).repeat_interleave(self.num_generations)
        # advantages = (rewards - mean_rewards) / (std_rewards + 1e-4)
        # advantages = torch.clamp(advantages, -5, 5)
        
        
        # self._log_step(images, prompts_text, advantages, completions)

        # # CoT loss computation
        # cot_loss, mean_kl_cot, completion_mask = self._compute_cot_loss(
        #     model, prompt_completion_ids, completion_ids, advantages, prompt_length
        # )
        
        # # Diffusion loss computation
        # model_pred = self._compute_diffusion_loss(
        #     model, completions, prev_latents, pred_latents, ts, "t2i_queries", 1
        # )
        
        # with torch.no_grad():
        #     ref_model_pred = self._compute_diffusion_loss(
        #         self.ref_model, completions, prev_latents, pred_latents, ts, "t2i_queries", 1
        #     )
        
        # # Compute log probs and KL for diffusion
        # _, log_prob_diff, mean_diff, std_diff = compute_log_prob(
        #     model_pred, self.scheduler, prev_latents, pred_latents, ts
        # )
        # _, _, mean_ref_diff, std_ref_diff = compute_log_prob(
        #     ref_model_pred, self.scheduler, prev_latents, pred_latents, ts
        # )
        
        # kl_diff = (mean_diff - mean_ref_diff)**2 / (2 * std_diff**2)
        # kl_diff = kl_diff.mean(dim=tuple(range(1, kl_diff.ndim)))
        
        # # Diffusion GRPO loss
        # advantages_diff = advantages.repeat_interleave(
        #     self.diffusion_config["num_inference_steps"], dim=0
        # )
        # ratio_diff = torch.exp(log_prob_diff - diff_log_probs_traj)
        # assert (ratio_diff == 1).all(), f"{ratio_diff}"

        # unclipped_loss_diff = -advantages_diff * ratio_diff
        # clipped_loss_diff = -advantages_diff * torch.clamp(
        #     ratio_diff, 1.0 - 1e-4, 1.0 + 1e-4
        # )
        # diff_loss = torch.mean(torch.maximum(unclipped_loss_diff, clipped_loss_diff))
        # diff_loss = diff_loss + self.beta * kl_diff.mean()
        
        # Combined loss
        loss = cot_loss + diff_loss
        
        # Logging
        completion_length = self.accelerator.gather_for_metrics(
            completion_mask.sum(1)
        ).float().mean().item()
        
        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
        self._metrics["cot_loss"].append(self.accelerator.gather_for_metrics(cot_loss).mean().item())
        self._metrics["cot_kl"].append(self.accelerator.gather_for_metrics(mean_kl_cot).mean().item())
        self._metrics["diff_loss"].append(self.accelerator.gather_for_metrics(diff_loss).mean().item())
        self._metrics["diff_kl"].append(self.accelerator.gather_for_metrics(kl_diff).mean().item())
        self._metrics["completion_length"].append(completion_length)
        
        for i, (func_name, _, _) in enumerate(self.reward_funcs):
            self._metrics[f"reward/{func_name}"].append(
                self.accelerator.gather_for_metrics(
                    rewards_per_func[:, i]
                ).mean().item()
            )
        
        return loss