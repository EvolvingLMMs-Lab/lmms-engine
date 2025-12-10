"""
Bagel GRPO Trainer for lmms-engine framework.

This trainer implements Group Relative Policy Optimization (GRPO) for Bagel model training.
It follows the training loop: Sampling -> Reward Computation -> Advantage Estimation -> Training.
"""

import gc
import hashlib
import os
import random
import tempfile
import time
from collections import defaultdict
from concurrent import futures
from functools import partial
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from accelerate.utils import send_to_device
from loguru import logger
from PIL import Image
from torch.distributed.fsdp import register_fsdp_forward_method
from torch.utils.data import Dataset, DistributedSampler, IterableDataset
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
from transformers.trainer_pt_utils import DistributedLengthGroupedSampler
from transformers.trainer_utils import seed_worker

import lmms_engine.models.utils as model_utils
import lmms_engine.parallel.process_group_manager as pgm
from lmms_engine.parallel.parallelize import MODEL_TO_PARALLEL_METHOD, apply_parallelize
from lmms_engine.train.config import TrainingArguments
from lmms_engine.train.fsdp2.fsdp2_trainer import FSDP2SFTTrainer
from lmms_engine.train.registry import TRAINER_REGISTER
from lmms_engine.utils import TrainUtilities
from lmms_engine.utils.fsdp2_utils import (
    apply_fsdp2,
    fsdp2_clip_grad_norm_,
    fsdp2_load_full_state_dict,
    get_constant_schedule,
    get_cosine_schedule_with_warmup,
    get_wsd_schedule_with_warmup,
)
from lmms_engine.utils.profiler import StepProfiler
from lmms_engine.utils.tracking import Tracking

DatasetType = Union[Dataset, IterableDataset]


def create_generators(prompts: List[str], base_seed: int = 42) -> List[torch.Generator]:
    """Create generators for each prompt based on stable hash."""
    generators = []
    for prompt in prompts:
        hash_digest = hashlib.sha256(prompt.encode()).digest()
        prompt_hash_int = int.from_bytes(hash_digest[:4], "big")
        seed = (base_seed + prompt_hash_int) % (2**31)
        gen = torch.Generator().manual_seed(seed)
        generators.append(gen)
    return generators


def calculate_zero_std_ratio(prompts: List[str], gathered_rewards: Dict[str, np.ndarray]) -> tuple:
    """Calculate the proportion of unique prompts whose reward standard deviation is zero."""
    prompt_array = np.array(prompts)
    unique_prompts, inverse_indices, counts = np.unique(prompt_array, return_inverse=True, return_counts=True)

    grouped_rewards = gathered_rewards["ori_avg"][np.argsort(inverse_indices)]
    split_indices = np.cumsum(counts)[:-1]
    reward_groups = np.split(grouped_rewards, split_indices)

    prompt_std_devs = np.array([np.std(group) for group in reward_groups])
    zero_std_count = np.count_nonzero(prompt_std_devs == 0)
    zero_std_ratio = zero_std_count / len(prompt_std_devs)

    return zero_std_ratio, prompt_std_devs.mean()


@TRAINER_REGISTER.register("bagel_grpo_trainer")
class BagelGRPOTrainer(FSDP2SFTTrainer):
    """
    GRPO Trainer for Bagel model.

    This trainer implements the GRPO training loop:
    1. Sampling: Generate images using the current policy
    2. Reward Computation: Compute rewards for generated images
    3. Advantage Estimation: Calculate advantages using per-prompt or global normalization
    4. Training: Update policy using PPO-style loss with KL penalty
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        args: TrainingArguments,
        train_dataset: DatasetType,
        eval_dataset: DatasetType = None,
        processing_class=None,
        data_collator=None,
        **kwargs,
    ) -> None:
        """
        Initialize BagelGRPOTrainer.

        Required arguments (passed by TrainRunner):
            model: The model to train
            args: TrainingArguments
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            processing_class: Processor/tokenizer
            data_collator: Data collator

        Optional arguments (passed via extra_kwargs in config):
            grpo_config: GRPO configuration dict or object
            inferencer: Optional inferencer instance
            reward_fn: Optional reward function
            eval_reward_fn: Optional evaluation reward function
        """
        super().__init__(model, args, train_dataset, eval_dataset, processing_class, data_collator)

        # Get grpo_config from multiple possible sources:
        # 1. Directly from kwargs (if runner passes extra_kwargs)
        # 2. From train_dataset.config.extra_kwargs (workaround if runner doesn't pass extra_kwargs)
        grpo_config = kwargs.pop("grpo_config", None)

        # Fallback: try to get from train_dataset.config.extra_kwargs
        if grpo_config is None and hasattr(train_dataset, "config"):
            dataset_extra_kwargs = getattr(train_dataset.config, "extra_kwargs", None)
            if dataset_extra_kwargs and isinstance(dataset_extra_kwargs, dict):
                grpo_config = dataset_extra_kwargs.get("grpo_config", None)

        # Convert dict config to object if needed
        if grpo_config is not None and isinstance(grpo_config, dict):
            self.grpo_config = self._dict_to_config(grpo_config)
        else:
            self.grpo_config = grpo_config

        # Validate grpo_config
        if self.grpo_config is None:
            raise ValueError(
                "grpo_config must be provided for BagelGRPOTrainer. "
                "Add it to dataset_config.extra_kwargs.grpo_config in your YAML config."
            )

        # GRPO-specific components - can be passed directly or initialized from config
        self.inferencer = kwargs.pop("inferencer", None)
        self.reward_fn = kwargs.pop("reward_fn", None)
        self.eval_reward_fn = kwargs.pop("eval_reward_fn", None)

        # If not provided, try to initialize from kwargs or model attributes
        if self.inferencer is None or self.reward_fn is None:
            self._initialize_grpo_components(kwargs)

        # Executor for async reward computation
        self.executor = futures.ThreadPoolExecutor(max_workers=8)

        # Stat tracker for per-prompt normalization
        if hasattr(self.grpo_config, "per_prompt_stat_tracking") and self.grpo_config.per_prompt_stat_tracking:
            from lmms_engine.utils.tracking import PerPromptStatTracker

            self.stat_tracker = PerPromptStatTracker(self.grpo_config.sample.global_std)
        else:
            self.stat_tracker = None

        # Reference model for KL penalty (if beta > 0)
        self.language_model_ref = None
        if (
            hasattr(self.grpo_config, "train")
            and hasattr(self.grpo_config.train, "beta")
            and self.grpo_config.train.beta > 0
        ):
            # Reference model will be set up in prepare_model
            pass

    def _initialize_grpo_components(self, kwargs):
        """
        Initialize GRPO components (inferencer, reward_fn) from model or kwargs.

        This allows the trainer to work with lmms-engine's standard initialization flow.
        """
        import torch.distributed as dist

        # Get device
        device = f"cuda:{dist.get_rank()}" if dist.is_initialized() else "cuda:0"

        # Initialize inferencer if not provided
        if self.inferencer is None:
            # Try to get from kwargs
            if "inferencer" in kwargs:
                self.inferencer = kwargs["inferencer"]
            else:
                # Try to initialize from model
                self.inferencer = self._create_inferencer_from_model(device)

        # Initialize reward functions if not provided
        if self.reward_fn is None:
            if "reward_fn" in kwargs:
                self.reward_fn = kwargs["reward_fn"]
            else:
                # Get reward function config from grpo_config
                # reward_fn can be:
                # 1. A dict like {"pickscore": 1.0} or {"geneval": 1.0, "aesthetic": 0.5}
                # 2. A string like "multi_score" (for backward compatibility, will use empty dict)
                reward_fn_config = getattr(self.grpo_config, "reward_fn", {})

                # Convert SimpleNamespace to dict if needed (from _dict_to_config)
                from types import SimpleNamespace

                if isinstance(reward_fn_config, SimpleNamespace):
                    reward_fn_config = {k: v for k, v in reward_fn_config.__dict__.items()}
                    logger.info(f"Converted SimpleNamespace reward_fn_config to dict: {reward_fn_config}")

                # If it's a string, treat it as a single reward with weight 1.0
                if isinstance(reward_fn_config, str):
                    reward_fn_config = {reward_fn_config: 1.0}
                # If it's not a dict, try to convert it
                elif not isinstance(reward_fn_config, dict):
                    logger.warning(
                        f"reward_fn config should be a dict, got {type(reward_fn_config)}. Converting to dict."
                    )
                    reward_fn_config = {}

                try:
                    import lmms_engine.utils.rewards.rewards

                    # Always use multi_score to handle dict config
                    self.reward_fn = lmms_engine.utils.rewards.rewards.multi_score(device, reward_fn_config)
                    logger.info(f"Initialized reward function with config: {reward_fn_config}")
                except (ImportError, AttributeError) as e:
                    logger.warning(f"Could not import reward function: {e}. Reward function must be provided.")
                    raise

        if self.eval_reward_fn is None:
            if "eval_reward_fn" in kwargs:
                self.eval_reward_fn = kwargs["eval_reward_fn"]
            else:
                self.eval_reward_fn = self.reward_fn  # Use same function for eval

    def _create_inferencer_from_model(self, device):
        """
        Create inferencer from model and processing_class.

        This assumes the model is a Bagel model and processing_class is a tokenizer.
        """
        try:
            from lmms_engine.models.bagel.data_utils import add_special_tokens
            from lmms_engine.models.bagel.inferencer import InterleaveInferencer
            from lmms_engine.models.bagel.transforms import ImageTransform
        except ImportError:
            raise ImportError("Could not import Bagel modules. Please check your installation.")

        # Get tokenizer from processing_class
        # BagelDataProcessor stores tokenizer in self.processor (not self.tokenizer which has a bug)
        if hasattr(self.processing_class, "processor"):
            tokenizer = self.processing_class.processor
        else:
            tokenizer = self.processing_class

        # Add special tokens
        tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

        # Get VAE model from Bagel model
        if hasattr(self.model, "vae_model"):
            vae_model = self.model.vae_model
        else:
            logger.warning("Could not find vae_model in model. Inferencer may not work correctly.")
            vae_model = None

        # Create transforms
        vae_transform = ImageTransform(512, 256, 8)
        vit_transform = ImageTransform(490, 112, 7)

        # Create inferencer
        inferencer = InterleaveInferencer(
            model=self.model,
            vae_model=vae_model,
            tokenizer=tokenizer,
            vae_transform=vae_transform,
            vit_transform=vit_transform,
            new_token_ids=new_token_ids,
        )

        return inferencer

    def _dict_to_config(self, config_dict):
        """
        Convert dictionary config to a simple object for easier access.

        This allows the config to be passed as a dict from YAML but accessed as attributes.
        """
        from types import SimpleNamespace

        def dict_to_namespace(d):
            if isinstance(d, dict):
                return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
            elif isinstance(d, list):
                return [dict_to_namespace(item) for item in d]
            else:
                return d

        return dict_to_namespace(config_dict)

    def prepare_model(self):
        """
        Prepare model for GRPO training.
        """
        # Keep the runtime patch to avoid DTensor/Tensor RMSNorm issues without touching model code
        self._patch_qwen2_rmsnorm_for_dtensor()

        # Ensure the layer class list includes the needed modules for FSDP wrapping
        try:
            cls_list = self.args.fsdp_config.get("transformer_layer_cls_to_wrap", [])
            if cls_list is None:
                cls_list = []
            if isinstance(cls_list, str):
                cls_list = [cls_list]
            for name in ["Qwen2MoTDecoderLayer", "AutoEncoder"]:
                if name not in cls_list:
                    cls_list.append(name)
            self.args.fsdp_config["transformer_layer_cls_to_wrap"] = cls_list
        except Exception as e:
            logger.warning(f"Failed to update transformer_layer_cls_to_wrap: {e}")

        # Use base FSDP2SFTTrainer prepare_model to wrap the model
        super().prepare_model()

        # After super, self.model is FSDP-wrapped
        self.fsdp2_model = self.model

        # Update inferencer references to the wrapped model/vae_model
        if hasattr(self, "inferencer") and self.inferencer is not None:
            if hasattr(self.inferencer, "model"):
                self.inferencer.model = self.model
            if hasattr(self.model, "vae_model") and hasattr(self.inferencer, "vae_model"):
                self.inferencer.vae_model = self.model.vae_model

        # Register FSDP forward methods similar to uni_fsdp_engine
        try:
            register_fsdp_forward_method(self.model, "forward_cache_update_vae")
            register_fsdp_forward_method(self.model, "forward_cache_update_vit")
            register_fsdp_forward_method(self.model, "forward_cache_update_text")
            register_fsdp_forward_method(self.model, "generate_image")
            register_fsdp_forward_method(self.model, "_forward_flow")
            if hasattr(self.model, "vae_model") and self.model.vae_model is not None:
                register_fsdp_forward_method(self.model.vae_model, "decode")
                register_fsdp_forward_method(self.model.vae_model, "encode")
        except Exception as e:
            logger.warning(f"Failed to register FSDP forward methods: {e}")

    def _patch_qwen2_rmsnorm_for_dtensor(self):
        """
        Runtime patch Qwen2RMSNorm.forward to handle DTensor weights (FSDP2) mixing with local tensors.
        This avoids editing model source files while preventing runtime errors.
        """
        try:
            from lmms_engine.models.bagel.qwen2.modeling_qwen2 import Qwen2RMSNorm
        except Exception as e:
            logger.warning(f"Could not import Qwen2RMSNorm for patching: {e}")
            return

        if getattr(Qwen2RMSNorm, "_patched_for_dtensor", False):
            return

        def _patched_forward(self, hidden_states):
            input_dtype = hidden_states.dtype
            hidden_states_fp32 = hidden_states.to(torch.float32)
            variance = hidden_states_fp32.pow(2).mean(-1, keepdim=True)
            hidden_states_fp32 = hidden_states_fp32 * torch.rsqrt(variance + self.variance_epsilon)

            weight = self.weight
            try:
                return weight * hidden_states_fp32.to(input_dtype)
            except RuntimeError as e:
                # Handle DTensor/local tensor mix by gathering full weight if needed
                if "mixed torch.Tensor and DTensor" in str(e) or "must match the size of tensor" in str(e):
                    try:
                        from torch.distributed.tensor import DTensor

                        if isinstance(weight, DTensor):
                            weight_full = weight.full_tensor()
                            return weight_full * hidden_states_fp32.to(input_dtype)
                    except Exception:
                        pass
                raise e

        Qwen2RMSNorm.forward = _patched_forward
        Qwen2RMSNorm._patched_for_dtensor = True
        logger.info("Patched Qwen2RMSNorm.forward at runtime to handle DTensor weights.")

    def prepare_optimizer(self):
        """
        Prepare optimizer for GRPO training.

        Only optimize parameters from language_model that require gradients.
        This matches train_bagel.py which only optimizes transformer parameters.
        """
        if hasattr(self.model, "language_model"):
            # Only optimize trainable parameters from language_model
            trainable_params = [p for p in self.model.language_model.parameters() if p.requires_grad]
            logger.info(f"Number of trainable parameters: {sum(p.numel() for p in trainable_params)}")
        else:
            # Fallback to all model parameters
            trainable_params = [p for p in self.fsdp2_model.parameters() if p.requires_grad]

        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
        )

    def compute_loss(self, batch):
        """Not used in GRPO training, but kept for compatibility."""
        # GRPO uses custom training_step instead
        raise NotImplementedError("GRPO trainer uses custom training_step")

    def training_step(self, batch):
        """
        GRPO training step.

        This is called for each training batch, but in GRPO we need to:
        1. Sample images
        2. Compute rewards
        3. Estimate advantages
        4. Train on the collected samples
        """
        # This method is overridden by the custom train() method
        # We don't use the standard training_step in GRPO
        raise NotImplementedError("Use custom train() method for GRPO")

    def _sampling_phase(self, epoch: int, train_iter):
        """
        Sampling phase: Generate images using current policy.

        Returns:
            samples: List of dicts containing latents, log_probs, timesteps, rewards (futures)
        """
        self.fsdp2_model.eval()
        samples = []

        num_batches_per_epoch = self.grpo_config.sample.num_batches_per_epoch
        train_batch_size = self.grpo_config.sample.train_batch_size

        # DEBUG: Log sampling configuration
        if dist.get_rank() == 0:
            logger.info(
                f"[DEBUG] _sampling_phase: epoch={epoch}, num_batches_per_epoch={num_batches_per_epoch}, train_batch_size={train_batch_size}"
            )

        if num_batches_per_epoch <= 0:
            logger.error(
                f"[DEBUG] CRITICAL: num_batches_per_epoch={num_batches_per_epoch} is invalid! Sampling will be skipped!"
            )

        for i in tqdm(
            range(num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=dist.get_rank() != 0,
            position=0,
        ):
            # Get prompts from dataloader
            if hasattr(train_iter, "set_epoch"):
                train_iter.set_epoch(epoch * num_batches_per_epoch + i)

            try:
                prompts, prompt_metadata = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_dataloader)
                prompts, prompt_metadata = next(train_iter)

            # Tokenize prompts
            # BagelDataProcessor stores tokenizer in self.processor (not self.tokenizer which has a bug)
            if hasattr(self.processing_class, "processor"):
                tokenizer = self.processing_class.processor
            else:
                # Fallback: assume processing_class is tokenizer
                tokenizer = self.processing_class

            prompt_ids = tokenizer(
                prompts,
                padding="max_length",
                max_length=256,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(self.fsdp2_model.device)

            # Calculate and accumulate tokens for this batch
            # In sampling phase, we generate 1 image per prompt
            batch_tokens = self._calculate_tokens_for_batch(prompt_ids, num_images=1)
            self.total_tokens += batch_tokens

            # Create generators for reproducible sampling
            generators = create_generators(prompts, base_seed=42)

            # Sample images
            images = []
            latents = []
            log_probs = []
            timesteps = []

            # Use autocast if configured
            autocast_dtype = torch.bfloat16 if self.args.bf16 else torch.float16
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                for idx, prompt in enumerate(prompts):
                    if self.grpo_config.sample.same_latent:
                        generator = generators[idx : idx + 1]
                    else:
                        generator = None

                    with torch.no_grad():
                        output_dict = self.inferencer(
                            text=prompt,
                            noise_level=self.grpo_config.sample.noise_level,
                            grpo_config=self.grpo_config,
                            accelerator=None,  # Not using accelerate in this context
                            num_timesteps=self.grpo_config.sample.num_steps,
                            cfg_text_scale=self.grpo_config.sample.guidance_scale,
                            generators=generator,
                            **self._get_inference_hyperparams(),
                        )

                    # Image is already tensor (C, H, W) in [0, 1], same as flow_grpo
                    images.append(output_dict["image"])
                    latents.append(output_dict["all_latents"])
                    log_probs.append(output_dict["all_log_probs"])
                    timesteps.append(output_dict["timesteps"])

            # Stack tensors
            stacked_inner_latents = [torch.stack(inner_list, dim=0) for inner_list in latents]
            latents = torch.stack(stacked_inner_latents, dim=0)
            stacked_inner_log_probs = [torch.stack(inner_list, dim=0) for inner_list in log_probs]
            log_probs = torch.stack(stacked_inner_log_probs, dim=0)
            timesteps = torch.stack(timesteps, dim=0)
            images = torch.stack(images, dim=0)

            # Debug: log shapes before reward computation
            logger.info(f"Sampling batch {i}: images shape: {images.shape}, prompts len: {len(prompts)}")
            logger.info(f"Sampling batch {i}: prompts: {prompts[:2] if len(prompts) > 0 else 'empty'}")
            if prompt_metadata:
                logger.info(
                    f"Sampling batch {i}: prompt_metadata type: {type(prompt_metadata)}, len: {len(prompt_metadata) if hasattr(prompt_metadata, '__len__') else 'N/A'}"
                )

            # Compute rewards asynchronously
            logger.info(
                f"Submitting reward computation for batch {i}: images shape {images.shape}, {len(prompts)} prompts"
            )
            rewards = self.executor.submit(self.reward_fn, images, prompts, prompt_metadata, only_strict=True)
            time.sleep(0)  # Yield to start reward computation

            samples.append(
                {
                    "prompt_ids": prompt_ids,
                    "timesteps": timesteps,
                    "latents": latents[:, :-1],  # Each entry is the latent before timestep t
                    "prev_latents": latents[:, 1:],  # Each entry is the latent after timestep t
                    "log_probs": log_probs,
                    "rewards": rewards,
                }
            )

        # Wait for all rewards to be computed
        for sample_idx, sample in enumerate(
            tqdm(
                samples,
                desc="Waiting for rewards",
                disable=dist.get_rank() != 0,
                position=0,
            )
        ):
            try:
                rewards, reward_metadata = sample["rewards"].result()
            except Exception as e:
                logger.error(f"Error getting reward result for sample {sample_idx}: {e}", exc_info=True)
                batch_size = sample["prompt_ids"].shape[0]
                rewards = {"avg": [0.0] * batch_size}
                reward_metadata = {}

            # Debug reward shape
            batch_size = sample["prompt_ids"].shape[0]
            logger.debug(f"Sample {sample_idx}: batch_size={batch_size}, rewards keys: {rewards.keys()}")

            if "avg" in rewards:
                avg_val = rewards["avg"]
                # Handle different types
                if isinstance(avg_val, torch.Tensor):
                    avg_len = avg_val.shape[0] if avg_val.dim() > 0 else 1
                    avg_val = avg_val.cpu().tolist() if avg_val.dim() > 0 else [avg_val.item()]
                elif isinstance(avg_val, (list, tuple)):
                    avg_len = len(avg_val)
                else:
                    avg_len = 1
                    avg_val = [float(avg_val)]

                logger.debug(
                    f"Sample {sample_idx}: rewards['avg'] type: {type(rewards['avg'])}, len: {avg_len}, batch_size: {batch_size}"
                )

                if avg_len != batch_size:
                    logger.error(
                        f"Reward length mismatch! rewards['avg'] len: {avg_len}, batch_size: {batch_size}, "
                        f"rewards['avg'] type: {type(rewards['avg'])}, value: {rewards['avg']}"
                    )
                    # Attempt to fix if it's 0 vs N (maybe empty list?)
                    if avg_len == 0:
                        logger.warning("Empty rewards received. Filling with zeros.")
                        rewards["avg"] = [0.0] * batch_size
                    elif avg_len == 1 and batch_size > 1:
                        logger.warning(f"Single reward for batch of {batch_size}. Repeating reward.")
                        rewards["avg"] = avg_val * batch_size
                    else:
                        logger.warning(f"Cannot fix mismatch. Using zeros.")
                        rewards["avg"] = [0.0] * batch_size
                else:
                    rewards["avg"] = avg_val
            else:
                logger.warning(f"No 'avg' in rewards. Keys: {rewards.keys()}")
                rewards["avg"] = [0.0] * batch_size

            # Convert all reward values to tensors
            sample["rewards"] = {}
            for key, value in rewards.items():
                if isinstance(value, torch.Tensor):
                    sample["rewards"][key] = value.to(device=self.fsdp2_model.device).float()
                elif isinstance(value, (list, tuple)):
                    sample["rewards"][key] = torch.as_tensor(value, device=self.fsdp2_model.device).float()
                else:
                    sample["rewards"][key] = torch.tensor([float(value)], device=self.fsdp2_model.device).float()

            logger.debug(
                f"Sample {sample_idx}: Final rewards shape: {[(k, v.shape if isinstance(v, torch.Tensor) else len(v)) for k, v in sample['rewards'].items()]}"
            )

        # Collate samples
        samples_collated = {
            k: torch.cat([s[k] for s in samples], dim=0)
            if not isinstance(samples[0][k], dict)
            else {sub_key: torch.cat([s[k][sub_key] for s in samples], dim=0) for sub_key in samples[0][k]}
            for k in samples[0].keys()
        }

        # DEBUG: Check if samples were collected
        if dist.get_rank() == 0:
            logger.info(f"[DEBUG] Sampling phase completed: collected {len(samples)} batches")
            if len(samples) == 0:
                logger.error("[DEBUG] CRITICAL: No samples collected! Check num_batches_per_epoch and dataloader.")
            else:
                for key in samples_collated:
                    if isinstance(samples_collated[key], torch.Tensor):
                        logger.info(f"[DEBUG] samples_collated['{key}'] shape: {samples_collated[key].shape}")
                    elif isinstance(samples_collated[key], dict):
                        for sub_key, sub_val in samples_collated[key].items():
                            if isinstance(sub_val, torch.Tensor):
                                logger.info(f"[DEBUG] samples_collated['{key}']['{sub_key}'] shape: {sub_val.shape}")

        # Get collated rewards for logging (before adding ori_avg)
        collated_rewards = samples_collated["rewards"]

        # Get rewards for the last batch for logging
        # Note: samples is a list of dicts, each dict contains rewards for that batch
        last_batch_rewards = samples[-1]["rewards"]

        return samples_collated, images, prompts, last_batch_rewards

    def _compute_advantages(self, samples: Dict, prompts: List[str], tokenizer):
        """
        Compute advantages from rewards.

        Returns:
            advantages: Tensor of advantages
        """
        # Gather rewards across processes (matching train_bagel.py line 745-746)
        gathered_rewards = {}
        for key, value in samples["rewards"].items():
            gathered_value = self._gather_tensor(value)
            gathered_rewards[key] = gathered_value.cpu().numpy()

        # Debug gathered rewards shape
        if "avg" in gathered_rewards:
            logger.info(f"Gathered rewards['avg'] shape: {gathered_rewards['avg'].shape}")

        # Log rewards (matching train_bagel.py line 747-755)
        if dist.get_rank() == 0:
            reward_metrics = {
                "epoch": self.global_step // self.steps_per_epoch if hasattr(self, "steps_per_epoch") else 0,
                **{
                    f"reward_{key}": float(
                        value.mean().item() if isinstance(value.mean(), torch.Tensor) else value.mean()
                    )
                    for key, value in gathered_rewards.items()
                    if "_strict_accuracy" not in key and "_accuracy" not in key
                },
            }
            if hasattr(self, "tracking"):
                self.tracking.log(reward_metrics, step=self.global_step)

        # Compute advantages
        if self.stat_tracker is not None:
            # Per-prompt normalization
            prompt_ids = self._gather_tensor(samples["prompt_ids"]).cpu().numpy()
            prompts_decoded = tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
            advantages = self.stat_tracker.update(prompts_decoded, gathered_rewards["avg"])

            # Check if advantages are all zeros (due to zero std)
            if isinstance(advantages, np.ndarray):
                if np.all(advantages == 0) or np.std(advantages) < 1e-6:
                    logger.warning(
                        "All advantages are zero or have zero std in per-prompt normalization. Using raw rewards minus mean."
                    )
                    advantages = gathered_rewards["avg"] - gathered_rewards["avg"].mean()

            # Log stat tracker metrics
            if dist.get_rank() == 0:
                group_size, trained_prompt_num = self.stat_tracker.get_stats()
                zero_std_ratio, reward_std_mean = calculate_zero_std_ratio(prompts_decoded, gathered_rewards)
                if hasattr(self, "tracking"):
                    # Convert numpy scalars to Python scalars for wandb
                    self.tracking.log(
                        {
                            "group_size": float(group_size)
                            if isinstance(group_size, (np.integer, np.floating))
                            else group_size,
                            "trained_prompt_num": int(trained_prompt_num)
                            if isinstance(trained_prompt_num, (np.integer, np.floating))
                            else trained_prompt_num,
                            "zero_std_ratio": float(zero_std_ratio)
                            if isinstance(zero_std_ratio, (np.integer, np.floating))
                            else zero_std_ratio,
                            "reward_std_mean": float(reward_std_mean)
                            if isinstance(reward_std_mean, (np.integer, np.floating))
                            else reward_std_mean,
                        },
                        step=self.global_step,
                    )

            self.stat_tracker.clear()
        else:
            # Global normalization
            reward_std = gathered_rewards["avg"].std()
            if reward_std == 0 or reward_std < 1e-6:
                logger.warning(f"Reward std is zero or very small ({reward_std}), using raw rewards as advantages")
                advantages = gathered_rewards["avg"] - gathered_rewards["avg"].mean()
            else:
                advantages = (gathered_rewards["avg"] - gathered_rewards["avg"].mean()) / (reward_std + 1e-4)

        # Ungather advantages to keep only entries for this process
        advantages = torch.as_tensor(advantages)
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        advantages = advantages.reshape(world_size, -1, advantages.shape[-1])[rank]
        advantages = advantages.to(self.fsdp2_model.device)

        # Debug: Check advantages for NaN/Inf
        if torch.isnan(advantages).any() or torch.isinf(advantages).any():
            logger.error(f"NaN/Inf detected in advantages! advantages: {advantages}, shape: {advantages.shape}")
            logger.error(f"  NaN count: {torch.isnan(advantages).sum()}, Inf count: {torch.isinf(advantages).sum()}")
            # Replace NaN/Inf with zeros
            advantages = torch.where(
                torch.isnan(advantages) | torch.isinf(advantages), torch.zeros_like(advantages), advantages
            )
        else:
            logger.info(
                f"Advantages stats: min={advantages.min()}, max={advantages.max()}, mean={advantages.mean()}, std={advantages.std()}"
            )

        return advantages

    def _force_inference_mode(self, module):
        """Recursively force ALL modules to be in inference mode (training=False).

        This is necessary because many classes in qwen2_navit.py use `if self.training:`
        to dispatch between forward_train and forward_inference. We need forward_inference
        for the inferencer to work correctly, even during training phase.

        Classes affected include:
        - Qwen2Attention, PackedAttentionMoT
        - Qwen2DecoderLayer, Qwen2MoTDecoderLayer, Qwen2DecoderLayerNaVIT
        - Qwen2Model, Qwen2ForCausalLM
        """
        # Force ALL modules to eval mode to ensure forward_inference is used
        # This is safe because:
        # 1. Dropout behavior is controlled by training flag, but we want inference path
        # 2. FSDP gradient sync works based on requires_grad, not training flag
        if module.training:
            module.training = False

        # Recurse into all children
        for child in module.children():
            self._force_inference_mode(child)

    def _training_phase(self, samples: Dict, advantages: torch.Tensor, tokenizer, epoch: int):
        """
        Training phase: Update policy using GRPO loss.

        Args:
            samples: Dict containing latents, log_probs, timesteps, etc.
            advantages: Computed advantages
            tokenizer: Tokenizer for decoding prompts
            epoch: Current epoch number
        """
        self.fsdp2_model.train()

        # Force inference mode for Qwen2 modules to ensure correct forward path
        # This is critical because Qwen2Model.forward checks self.training
        if hasattr(self.model, "language_model"):
            self._force_inference_mode(self.model.language_model)

        # Set training mode but disable dropout in some layers and force inference path for Qwen2Model
        # This matches train_bagel.py logic and handles FSDP/LoRA wrapping
        # (Legacy manual logic kept for safety but _force_inference_mode should cover it)
        if hasattr(self.model, "language_model"):
            # Start with the potentially FSDP-wrapped language model
            lm = self.model.language_model

            # Unwrap FSDP wrapper to get to the actual model (Qwen2ForCausalLM or PeftModel)
            # Note: We keep the FSDP wrapper in training=True for gradient sync
            if hasattr(lm, "module"):
                lm = lm.module
                lm.training = False  # Set the inner module to eval

            # Unwrap PeftModel if present
            # PeftModel usually puts the base model in .base_model or .model
            if hasattr(lm, "base_model"):
                lm = lm.base_model
            elif (
                hasattr(lm, "model")
                and not isinstance(lm.model, torch.nn.ModuleList)
                and not isinstance(lm.model, torch.nn.Sequential)
            ):
                # This handles cases where .model is the base model (like in some Peft implementations or Qwen2ForCausalLM)
                # But we need to be careful not to confuse with Qwen2Model inside Qwen2ForCausalLM which is also named .model
                # Qwen2ForCausalLM.model is Qwen2Model.
                pass

            # Now we expect lm to be Qwen2ForCausalLM (or similar) which has .model as Qwen2Model
            if hasattr(lm, "model"):
                qwen2_model = lm.model
                qwen2_model.training = False  # Force Qwen2Model to use forward_inference

                # Recursively set layers to eval
                if hasattr(qwen2_model, "layers"):
                    for layer in qwen2_model.layers:
                        # Layer might be FSDP wrapped
                        if hasattr(layer, "module"):
                            layer.module.training = False
                            if hasattr(layer.module, "self_attn"):
                                layer.module.self_attn.training = False
                        else:
                            layer.training = False
                            if hasattr(layer, "self_attn"):
                                layer.self_attn.training = False

        total_batch_size, num_timesteps = samples["timesteps"].shape
        num_inner_epochs = self.grpo_config.train.num_inner_epochs
        num_batches_per_epoch = self.grpo_config.sample.num_batches_per_epoch

        # DEBUG: Log training phase configuration
        if dist.get_rank() == 0:
            logger.info(
                f"[DEBUG] _training_phase: epoch={epoch}, total_batch_size={total_batch_size}, "
                f"num_timesteps={num_timesteps}, num_inner_epochs={num_inner_epochs}, "
                f"num_batches_per_epoch={num_batches_per_epoch}, "
                f"current_global_step={self.global_step}"
            )

            # Check if reshape will work
            if num_batches_per_epoch > 0:
                rebatch_size = total_batch_size // num_batches_per_epoch
                logger.info(f"[DEBUG] Rebatch size will be: {rebatch_size}")
                if rebatch_size == 0:
                    logger.error(
                        f"[DEBUG] CRITICAL: rebatch_size is 0! total_batch_size={total_batch_size}, "
                        f"num_batches_per_epoch={num_batches_per_epoch}"
                    )

        for inner_epoch in range(num_inner_epochs):
            # Rebatch for training
            samples_batched = {
                k: v.reshape(-1, total_batch_size // num_batches_per_epoch, *v.shape[1:]) for k, v in samples.items()
            }

            # Convert dict to list of dicts for easier iteration
            samples_batched = [dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())]

            info = defaultdict(list)

            for i, sample in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                disable=dist.get_rank() != 0,
                position=0,
            ):
                # Compute dtimesteps
                sample["dtimesteps"] = torch.cat(
                    [sample["timesteps"][:, :-1] - sample["timesteps"][:, 1:], sample["timesteps"][:, -1].unsqueeze(1)],
                    dim=1,
                )

                bs = sample["timesteps"].shape[0]
                prompts = tokenizer.batch_decode(sample["prompt_ids"], skip_special_tokens=True)

                # Calculate and accumulate tokens for this training batch
                # In training phase, we process 1 image per prompt (already generated in sampling)
                batch_tokens = self._calculate_tokens_for_batch(sample["prompt_ids"], num_images=1)
                self.total_tokens += batch_tokens

                for j in tqdm(
                    range(bs),
                    desc="Batch Size",
                    position=1,
                    leave=False,
                    disable=dist.get_rank() != 0,
                ):
                    # Create cur_sample, but keep advantages as a tensor (not indexed)
                    cur_sample = {}
                    for k, v in sample.items():
                        if k == "advantages":
                            # Keep advantages as tensor for broadcasting
                            # If v is [batch_size], we need to select v[j] but keep as [1] tensor
                            if v.dim() == 1:
                                cur_sample[k] = v[j : j + 1]  # Keep as [1] tensor
                            else:
                                cur_sample[k] = v[j]
                        else:
                            cur_sample[k] = v[j]

                    # Snap timesteps to the exact values in the training grid to avoid precision mismatch
                    # The issue: generate_image returns timesteps as Python floats -> tensor conversion,
                    # while generate_image_learn computes original_timesteps via torch.linspace directly.
                    # This causes floating-point precision differences, making (original_timesteps == timesteps[i])
                    # return empty, leading to dtimesteps[t_index] having size 0.
                    # Fix: compute the exact grid and snap sampled timesteps to their closest matches.
                    num_steps = self.grpo_config.sample.num_steps
                    timestep_shift = self.grpo_config.train.timestep_shift
                    device = cur_sample["timesteps"].device

                    # Compute the exact grid that generate_image_learn will use
                    grid = torch.linspace(1, 0, num_steps, device=device, dtype=torch.float32)
                    grid = timestep_shift * grid / (1 + (timestep_shift - 1) * grid)

                    # For each sampled timestep, find the closest match in the grid
                    sampled_timesteps = cur_sample["timesteps"]  # shape: [num_sde_window_steps]
                    # Expand dimensions for broadcasting: [num_sde_window_steps, 1] - [1, num_steps]
                    indices = torch.argmin(torch.abs(sampled_timesteps.unsqueeze(-1) - grid.unsqueeze(0)), dim=-1)
                    cur_sample["timesteps"] = grid[indices]

                    # Use autocast
                    autocast_dtype = torch.bfloat16 if self.args.bf16 else torch.float16
                    with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                        output_dict = self.inferencer(
                            text=prompts[j],
                            noise_level=self.grpo_config.sample.noise_level,
                            learn=True,
                            sample=cur_sample,
                            grpo_config=self.grpo_config,
                            accelerator=None,
                            optimizer=self.optimizer,
                            transformer=self.model.language_model,
                            num_timesteps=self.grpo_config.sample.num_steps,
                            cfg_text_scale=self.grpo_config.sample.guidance_scale,
                            **self._get_inference_hyperparams(),
                        )

                    info["clipfrac"].append(output_dict["clipfrac"])
                    info["clipfrac_gt_one"].append(output_dict["clipfrac_gt_one"])
                    info["clipfrac_lt_one"].append(output_dict["clipfrac_lt_one"])
                    info["policy_loss"].append(output_dict["policy_loss"])
                    info["kl_loss"].append(output_dict["kl_loss"])
                    info["loss"].append(output_dict["loss"])

                # Aggregate metrics when gradient sync happens
                if self._should_sync_gradients():
                    info_aggregated = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                    # Reduce across processes
                    for key in info_aggregated:
                        info_aggregated[key] = self._reduce_tensor(info_aggregated[key])

                    info_aggregated.update({"epoch": epoch, "inner_epoch": inner_epoch})

                    # Add total_tokens to metrics
                    # In distributed training, each process accumulates its own tokens
                    # We sum across all processes to get the global total
                    total_tokens_tensor = torch.tensor(
                        self.total_tokens, device=self.fsdp2_model.device, dtype=torch.long
                    )
                    dist.all_reduce(total_tokens_tensor, op=dist.ReduceOp.SUM)
                    # After all_reduce, each process has the sum of all processes' tokens
                    # Update self.total_tokens to the global sum for checkpoint saving
                    self.total_tokens = total_tokens_tensor.item()

                    # Log total_tokens (only from rank 0 to avoid duplicate)
                    if dist.get_rank() == 0:
                        from lmms_engine.utils import TrainUtilities

                        info_aggregated["train/total_tokens"] = TrainUtilities.format_tokens(self.total_tokens)

                        # Convert all torch.Tensor values to Python scalars for wandb
                        # wandb requires scalar values (int, float) not tensors
                        info_aggregated_for_logging = {}
                        for key, value in info_aggregated.items():
                            if isinstance(value, torch.Tensor):
                                # Convert tensor to scalar
                                if value.numel() == 1:
                                    info_aggregated_for_logging[key] = value.item()
                                else:
                                    # Skip multi-element tensors
                                    logger.warning(f"Skipping multi-element tensor for key {key}: shape {value.shape}")
                            elif isinstance(value, (int, float, np.integer, np.floating)):
                                # Already a scalar
                                info_aggregated_for_logging[key] = (
                                    float(value) if isinstance(value, (np.integer, np.floating)) else value
                                )
                            elif isinstance(value, str):
                                # Skip strings (wandb doesn't log them as metrics)
                                continue
                            else:
                                # Try to convert to float if possible
                                try:
                                    info_aggregated_for_logging[key] = float(value)
                                except (ValueError, TypeError):
                                    logger.warning(f"Skipping non-scalar value for key {key}: type {type(value)}")

                        # DEBUG: Log what we're about to send to wandb
                        logger.info(
                            f"[DEBUG] Logging to wandb at step={self.global_step}: "
                            f"policy_loss={info_aggregated_for_logging.get('policy_loss', 'N/A'):.6f}, "
                            f"kl_loss={info_aggregated_for_logging.get('kl_loss', 'N/A')}, "
                            f"loss={info_aggregated_for_logging.get('loss', 'N/A')}"
                        )

                        if hasattr(self, "tracking"):
                            self.tracking.log(info_aggregated_for_logging, step=self.global_step)
                        else:
                            logger.warning(f"[DEBUG] self.tracking does not exist!")

                    self.global_step += 1

                    # DEBUG: Log global_step update (on all ranks to verify synchronization)
                    logger.debug(f"[DEBUG] Rank {dist.get_rank()}: global_step updated to {self.global_step}")

                    info = defaultdict(list)

    def _get_inference_hyperparams(self) -> Dict:
        """Get inference hyperparameters from config."""
        return {
            "cfg_img_scale": 1.0,
            "cfg_interval": [0, 1.0],
            "timestep_shift": self.grpo_config.train.timestep_shift,
            "cfg_renorm_min": 0.0,
            "cfg_renorm_type": "global",
            "image_shapes": (self.grpo_config.resolution, self.grpo_config.resolution),
        }

    def _calculate_tokens_for_batch(self, prompt_ids: torch.Tensor, num_images: int = 1) -> int:
        """
        Calculate total tokens for a batch of samples.

        For Bagel GRPO, tokens include:
        1. Text tokens: prompt_ids length (excluding padding)
        2. Image tokens: num_image_tokens per image = (resolution / latent_downsample)^2
        3. Special tokens: start_of_image, end_of_image (2 per image)

        Args:
            prompt_ids: Tensor of shape [batch_size, seq_len] with tokenized prompts
            num_images: Number of images per prompt (default 1)

        Returns:
            Total number of tokens for this batch
        """
        # Calculate text tokens (excluding padding)
        # prompt_ids shape: [batch_size, seq_len]
        batch_size = prompt_ids.shape[0]
        seq_len = prompt_ids.shape[1]

        # Count non-padding tokens
        # Get pad_token_id from tokenizer if available
        pad_token_id = None
        if hasattr(self.processing_class, "processor"):
            tokenizer = self.processing_class.processor
        else:
            tokenizer = self.processing_class

        if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None:
            pad_token_id = tokenizer.pad_token_id
        elif hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
            # Some tokenizers use eos_token_id for padding
            pad_token_id = tokenizer.eos_token_id

        if pad_token_id is not None:
            # Count non-padding tokens
            non_padding_mask = prompt_ids != pad_token_id
            text_tokens = non_padding_mask.sum().item()
        else:
            # If no pad_token_id, count all tokens
            text_tokens = batch_size * seq_len

        # Calculate image tokens
        # For Bagel, image tokens = (resolution / latent_downsample)^2
        # Get latent_downsample from model config
        if hasattr(self.model, "latent_downsample"):
            latent_downsample = self.model.latent_downsample
        else:
            # Default: resolution 512, latent_downsample typically 8
            # So h = w = 512 / 8 = 64, num_image_tokens = 64 * 64 = 4096
            latent_downsample = self.grpo_config.resolution // 64  # Estimate: 512 / 64 = 8

        h = w = self.grpo_config.resolution // latent_downsample
        num_image_tokens_per_image = h * w

        # Special tokens: start_of_image and end_of_image (2 per image)
        special_tokens_per_image = 2

        # Total tokens per sample: text + (image_tokens + special_tokens) * num_images
        # Note: text_tokens is already the total for the batch, so we divide by batch_size
        tokens_per_sample = (text_tokens // batch_size) + (
            num_image_tokens_per_image + special_tokens_per_image
        ) * num_images

        # Total tokens for batch
        total_tokens = batch_size * tokens_per_sample

        return total_tokens

    def _gather_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Gather tensor across all processes."""
        gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, tensor)
        return torch.cat(gathered, dim=0)

    def _reduce_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reduce tensor across all processes."""
        reduced = tensor.clone()
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        reduced = reduced / dist.get_world_size()
        return reduced

    def _should_sync_gradients(self) -> bool:
        """Check if gradients should be synced (after accumulation)."""
        # This should align with gradient accumulation steps
        # For simplicity, assume sync after each step
        # In practice, this should check gradient accumulation counter
        return True

    def _auto_calculate_num_batches_per_epoch(self):
        """
        Automatically calculate num_batches_per_epoch based on dataset size.

        If num_batches_per_epoch is set to -1 in config, it will be auto-calculated
        to cover the entire dataset (or a reasonable subset).

        Returns:
            int or None: Number of batches per epoch if auto-calculation is needed, None otherwise
        """
        # Check if auto-calculation is requested (value is -1 or None)
        current_value = getattr(self.grpo_config.sample, "num_batches_per_epoch", None)

        # If explicitly set and not -1, don't auto-calculate
        if current_value is not None and current_value != -1:
            return None

        # Get dataset size
        if isinstance(self.train_dataset, IterableDataset):
            # For IterableDataset, we can't get length easily
            # Use a default value or warn
            logger.warning(
                "Cannot auto-calculate num_batches_per_epoch for IterableDataset. "
                "Using default value of 10. Please set num_batches_per_epoch manually."
            )
            return 10

        # Get dataset length
        dataset_size = len(self.train_dataset)

        # Get training parameters
        train_batch_size = getattr(self.grpo_config.sample, "train_batch_size", 6)
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        # Calculate total effective batch size (across all GPUs)
        effective_batch_size = train_batch_size * world_size

        # Calculate number of batches needed to cover the entire dataset
        # We use ceiling to ensure we cover all data
        num_batches = (dataset_size + effective_batch_size - 1) // effective_batch_size

        # Ensure at least 1 batch
        num_batches = max(1, num_batches)

        # Log the calculation
        if dist.get_rank() == 0:
            logger.info(
                f"Auto-calculating num_batches_per_epoch: "
                f"dataset_size={dataset_size}, train_batch_size={train_batch_size}, "
                f"world_size={world_size}, effective_batch_size={effective_batch_size}, "
                f"calculated num_batches_per_epoch={num_batches}"
            )

        return num_batches

    def train(self, resume_from_checkpoint: bool = False):
        """
        Main training loop for GRPO.

        The loop consists of:
        1. Sampling phase: Generate images
        2. Reward computation: Compute rewards (async, already done in sampling)
        3. Advantage estimation: Calculate advantages
        4. Training phase: Update policy
        """
        self.prepare_model()
        train_dataloader = self.prepare_dataloader(self.train_dataset, is_training=True)
        self.train_dataloader = train_dataloader

        # Auto-calculate num_batches_per_epoch if needed
        # IMPORTANT: Must happen BEFORE prepare_and_validate_config
        current_value = getattr(self.grpo_config.sample, "num_batches_per_epoch", None)
        logger.info(f"[DEBUG] Initial num_batches_per_epoch from config: {current_value}")

        if current_value is None or current_value == -1:
            auto_calculated = self._auto_calculate_num_batches_per_epoch()
            if auto_calculated is not None:
                self.grpo_config.sample.num_batches_per_epoch = auto_calculated
                logger.info(
                    f"[DEBUG] Auto-calculated num_batches_per_epoch: {auto_calculated} "
                    f"(dataset will be fully covered each epoch)"
                )
            else:
                # Fallback to a reasonable default if auto-calculation fails
                self.grpo_config.sample.num_batches_per_epoch = 10
                logger.warning(f"[DEBUG] Auto-calculation returned None, using default: 10")

        # Verify the final value
        final_value = self.grpo_config.sample.num_batches_per_epoch
        logger.info(f"[DEBUG] Final num_batches_per_epoch: {final_value}")

        if final_value <= 0:
            logger.error(f"[DEBUG] CRITICAL: num_batches_per_epoch is {final_value}, this will cause empty sampling!")
            # Force a reasonable default
            self.grpo_config.sample.num_batches_per_epoch = 10
            logger.warning(f"[DEBUG] Forced num_batches_per_epoch to 10")

        self.prepare_optimizer()

        # Validate config
        self.prepare_and_validate_config()

        # Prepare scheduler
        warmup_steps = (
            int(self.total_steps * self.args.warmup_ratio) if self.args.warmup_ratio > 0 else self.args.warmup_steps
        )
        self.prepare_scheduler(warmup_steps, self.total_steps)

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # Initialize tracking
        if rank == 0:
            self.tracking = Tracking(
                project_name=os.environ.get("WANDB_PROJECT", self.args.project),
                experiment_name=os.environ.get("WANDB_NAME", self.args.run_name),
                default_backend=self.default_backend,
                config=self.args,
            )

        # Initialize tokenizer
        # BagelDataProcessor stores tokenizer in self.processor (not self.tokenizer which has a bug)
        if hasattr(self.processing_class, "processor"):
            tokenizer = self.processing_class.processor
        else:
            tokenizer = self.processing_class

        # Initialize total_tokens (required by save_checkpoints)
        # This tracks total tokens processed during training (text + image tokens)
        # Will be accumulated during sampling and training phases
        self.total_tokens = 0

        # Resume from checkpoint
        if resume_from_checkpoint:
            checkpoints = [f for f in os.listdir(self.args.output_dir) if f.startswith("checkpoint")]
            checkpoints.sort(key=lambda x: int(x.split("-")[1]))
            latest_checkpoint = checkpoints[-1]
            self.load_checkpoints(
                os.path.join(self.args.output_dir, latest_checkpoint),
                int(latest_checkpoint.split("-")[1]),
            )
            # total_tokens will be loaded by load_checkpoints if present in checkpoint
            if not hasattr(self, "total_tokens"):
                self.total_tokens = 0
            start_epoch = int(latest_checkpoint.split("-")[1]) / self.steps_per_epoch
            start_epoch = int(start_epoch)
            self.global_step = int(latest_checkpoint.split("-")[1])
        else:
            start_epoch = 0
            self.global_step = 0

        logger.info(f"Training with {self.args.num_train_epochs} epochs, " f"{self.total_steps} total steps")

        train_iter = iter(self.train_dataloader)

        # Main training loop
        for epoch in range(start_epoch, self.args.num_train_epochs):
            # DEBUG: Log epoch start
            if dist.get_rank() == 0:
                logger.info(
                    f"[DEBUG] ===== Starting epoch {epoch}/{self.args.num_train_epochs} ===== "
                    f"global_step={self.global_step}"
                )

            # Evaluation (if needed)
            if (
                hasattr(self.grpo_config, "eval_freq")
                and epoch % self.grpo_config.eval_freq == 0
                and epoch > 0
                and self.eval_dataset is not None
            ):
                self._eval_phase(epoch)

            # Save checkpoint
            if self.should_save:
                output_dir = os.path.join(self.args.output_dir, f"checkpoint-{self.global_step}")
                self.save_checkpoints(
                    output_dir,
                    self.global_step,
                    total_limit=self.args.save_total_limit,
                )

            # Sampling phase
            samples, images, prompts, last_batch_rewards = self._sampling_phase(epoch, train_iter)

            # Add ori_avg for advantage computation (matching train_bagel.py line 742-743)
            samples["rewards"]["ori_avg"] = samples["rewards"]["avg"]
            samples["rewards"]["avg"] = samples["rewards"]["avg"].unsqueeze(-1)

            # Log sample images periodically (matching train_bagel.py line 713-741)
            if epoch % 5 == 0 and rank == 0:
                # Use last_batch_rewards for logging
                # Note: last_batch_rewards["avg"] is already correct for the last batch
                # We use last_batch_rewards which corresponds to images and prompts from the last batch
                rewards_for_logging = {"avg": last_batch_rewards["avg"]}
                self._log_sample_images(images, prompts, rewards_for_logging, epoch)

            # Compute advantages (matching train_bagel.py line 744-787)
            advantages = self._compute_advantages(samples, prompts, tokenizer)
            samples["advantages"] = advantages
            del samples["rewards"]  # Free memory (matching train_bagel.py line 789)

            # DEBUG: Check advantages before training
            if dist.get_rank() == 0:
                logger.info(
                    f"[DEBUG] Before training: advantages shape={advantages.shape}, "
                    f"min={advantages.min().item():.6f}, max={advantages.max().item():.6f}, "
                    f"mean={advantages.mean().item():.6f}, std={advantages.std().item():.6f}"
                )
                if advantages.std().item() < 1e-6:
                    logger.warning("[DEBUG] WARNING: advantages have near-zero std! Training may not be effective.")

            # Training phase
            self._training_phase(samples, advantages, tokenizer, epoch)

            # DEBUG: Log epoch completion
            if dist.get_rank() == 0:
                logger.info(f"[DEBUG] ===== Completed epoch {epoch} ===== global_step={self.global_step}")

            # Cleanup
            if (
                hasattr(self.args, "torch_empty_cache_steps")
                and self.args.torch_empty_cache_steps is not None
                and self.global_step % self.args.torch_empty_cache_steps == 0
            ):
                self.empty_cache()

        # Save final checkpoint
        output_dir = os.path.join(self.args.output_dir, f"checkpoint-{self.global_step}")
        self.save_checkpoints(output_dir, self.global_step, total_limit=self.args.save_total_limit)

    def _eval_phase(self, epoch: int):
        """
        Evaluation phase: Generate images on eval dataset and compute rewards.

        This follows the eval logic from train_bagel.py:
        1. Set model to eval mode
        2. Iterate through eval dataloader
        3. Generate images for each prompt
        4. Compute rewards asynchronously
        5. Gather results across processes
        6. Log images and rewards to wandb
        """
        import wandb

        if self.eval_dataset is None:
            logger.warning(f"Evaluation skipped at epoch {epoch}: no eval_dataset provided")
            return

        logger.info(f"[DEBUG] Starting evaluation at epoch {epoch}, global_step={self.global_step}")

        # Set model to eval mode
        self.fsdp2_model.eval()

        # Force inference mode for Qwen2 modules
        if hasattr(self.model, "language_model"):
            self._force_inference_mode(self.model.language_model)

        # Prepare eval dataloader if not already prepared
        if not hasattr(self, "eval_dataloader") or self.eval_dataloader is None:
            self.eval_dataloader = self.prepare_dataloader(self.eval_dataset, is_training=False)

        # Get tokenizer
        if hasattr(self.processing_class, "processor"):
            tokenizer = self.processing_class.processor
        else:
            tokenizer = self.processing_class

        # Collect all rewards across batches
        all_rewards = defaultdict(list)

        # Use autocast
        autocast_dtype = torch.bfloat16 if self.args.bf16 else torch.float16

        eval_num_steps = getattr(self.grpo_config.sample, "eval_num_steps", 50)
        eval_guidance_scale = getattr(self.grpo_config.sample, "eval_guidance_scale", 4.0)

        logger.info(f"[DEBUG] Eval config: eval_num_steps={eval_num_steps}, eval_guidance_scale={eval_guidance_scale}")

        # Store last batch for logging
        last_batch_images = None
        last_batch_prompts = None
        last_batch_rewards = None

        for batch_idx, batch in enumerate(
            tqdm(
                self.eval_dataloader,
                desc=f"Epoch {epoch}: evaluation",
                disable=dist.get_rank() != 0,
                position=0,
            )
        ):
            # Handle different batch formats
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                prompts, prompt_metadata = batch
            elif isinstance(batch, dict):
                prompts = batch.get("prompt", batch.get("text", []))
                prompt_metadata = batch.get("metadata", [{}] * len(prompts))
            else:
                logger.warning(f"Unexpected batch format: {type(batch)}")
                continue

            # Generate images for this batch
            images = []
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                for idx, prompt in enumerate(prompts):
                    with torch.no_grad():
                        output_dict = self.inferencer(
                            text=prompt,
                            noise_level=0,  # No noise for evaluation (same as flow_grpo)
                            grpo_config=self.grpo_config,
                            accelerator=None,
                            num_timesteps=eval_num_steps,
                            cfg_text_scale=eval_guidance_scale,
                            **self._get_inference_hyperparams(),
                        )
                    # Image is already tensor (C, H, W) in [0, 1], same as flow_grpo
                    images.append(output_dict["image"])

            # Stack images: (batch_size, 3, H, W)
            images = torch.stack(images, dim=0)

            # Compute rewards asynchronously
            rewards_future = self.executor.submit(
                self.eval_reward_fn, images, prompts, prompt_metadata, only_strict=False
            )
            time.sleep(0)  # Yield to start reward computation

            # Wait for rewards
            try:
                rewards, reward_metadata = rewards_future.result()
            except Exception as e:
                logger.error(f"Error computing eval rewards for batch {batch_idx}: {e}")
                rewards = {"avg": [0.0] * len(prompts)}

            # Gather rewards across processes
            for key, value in rewards.items():
                if isinstance(value, torch.Tensor):
                    value_tensor = value.to(self.fsdp2_model.device)
                else:
                    value_tensor = torch.as_tensor(value, device=self.fsdp2_model.device)
                rewards_gathered = self._gather_tensor(value_tensor).cpu().numpy()
                all_rewards[key].append(rewards_gathered)

            # Store last batch for logging
            last_batch_images = images
            last_batch_prompts = prompts
            last_batch_rewards = rewards

        # Concatenate all rewards
        all_rewards_concat = {key: np.concatenate(value) for key, value in all_rewards.items()}

        # Gather last batch images and prompts for logging
        if last_batch_images is not None and len(last_batch_images) > 0:
            # Gather images across processes
            images_gathered = self._gather_tensor(last_batch_images.to(self.fsdp2_model.device)).cpu().numpy()

            # Tokenize and gather prompts
            prompt_ids = tokenizer(
                last_batch_prompts,
                padding="max_length",
                max_length=256,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(self.fsdp2_model.device)
            prompt_ids_gathered = self._gather_tensor(prompt_ids).cpu().numpy()
            prompts_gathered = tokenizer.batch_decode(prompt_ids_gathered, skip_special_tokens=True)

            # Gather last batch rewards
            last_batch_rewards_gathered = {}
            for key, value in last_batch_rewards.items():
                if isinstance(value, torch.Tensor):
                    value_tensor = value.to(self.fsdp2_model.device)
                else:
                    value_tensor = torch.as_tensor(value, device=self.fsdp2_model.device)
                last_batch_rewards_gathered[key] = self._gather_tensor(value_tensor).cpu().numpy()
        else:
            images_gathered = np.array([])
            prompts_gathered = []
            last_batch_rewards_gathered = {}

        # Log to wandb (only on main process)
        if dist.get_rank() == 0:
            # Log mean rewards
            eval_metrics = {
                "eval/epoch": epoch,
                **{
                    f"eval/reward_{key}": float(np.mean(value[value != -10]))
                    for key, value in all_rewards_concat.items()
                    if len(value) > 0
                },
            }

            if hasattr(self, "tracking"):
                self.tracking.log(eval_metrics, step=self.global_step)
            logger.info(f"[DEBUG] Eval metrics: {eval_metrics}")

            # Log sample images (following flow_grpo's approach)
            if len(images_gathered) > 0:
                with tempfile.TemporaryDirectory() as tmpdir:
                    num_samples = min(25, len(images_gathered))
                    sample_indices = list(range(num_samples))

                    # Save images to temp directory
                    for idx, index in enumerate(sample_indices):
                        image = images_gathered[index]
                        # image shape: (C, H, W) -> (H, W, C), values in [0, 1]
                        pil = Image.fromarray((image.transpose(1, 2, 0) * 255).astype(np.uint8))
                        pil = pil.resize((self.grpo_config.resolution, self.grpo_config.resolution))
                        pil.save(os.path.join(tmpdir, f"{idx}.jpg"))

                    # Prepare prompts and rewards for caption
                    sampled_prompts = [prompts_gathered[i] for i in sample_indices if i < len(prompts_gathered)]
                    sampled_rewards = [
                        {k: last_batch_rewards_gathered[k][i] for k in last_batch_rewards_gathered}
                        for i in sample_indices
                        if i < len(prompts_gathered)
                    ]

                    # Create wandb.Image list (same format as flow_grpo)
                    eval_images = [
                        wandb.Image(
                            os.path.join(tmpdir, f"{idx}.jpg"),
                            caption=f"{prompt[:1000]} | "
                            + " | ".join(
                                f"{k}: {v:.2f}"
                                for k, v in reward_dict.items()
                                if isinstance(v, (int, float)) and v != -10
                            ),
                        )
                        for idx, (prompt, reward_dict) in enumerate(zip(sampled_prompts, sampled_rewards))
                    ]

                    # Log using wandb directly (same as flow_grpo)
                    # Use "eval_images" key so it shows separately from "images" in wandb Media panel
                    wandb.log(
                        {
                            "eval_images": eval_images,
                            **{
                                f"eval/reward_{key}": float(np.mean(value[value != -10]))
                                for key, value in all_rewards_concat.items()
                                if len(value) > 0
                            },
                        },
                        step=self.global_step,
                    )
                    logger.info(f"[DEBUG] Logged {len(eval_images)} eval_images to wandb")

        logger.info(f"[DEBUG] Evaluation completed at epoch {epoch}")

    def _log_sample_images(self, images: torch.Tensor, prompts: List[str], rewards: Dict, epoch: int):
        """Log sample images to wandb."""
        if not hasattr(self, "tracking"):
            return

        num_samples = min(15, len(images))
        if num_samples == 0:
            return

        sample_indices = random.sample(range(len(images)), num_samples)

        # Debug info for shape mismatch
        if "avg" in rewards:
            avg_rewards = rewards["avg"]
            rewards_len = (
                avg_rewards.shape[0]
                if isinstance(avg_rewards, torch.Tensor) and avg_rewards.dim() > 0
                else (1 if isinstance(avg_rewards, torch.Tensor) else len(avg_rewards))
            )
            if rewards_len < len(images):
                logger.warning(
                    f"Mismatch in logging: {len(images)} images but {rewards_len} rewards. Using 0.0 for missing rewards."
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            for idx, i in enumerate(sample_indices):
                image = images[i].cpu().numpy()
                pil = Image.fromarray((image.transpose(1, 2, 0) * 255).astype(np.uint8))
                pil = pil.resize((self.grpo_config.resolution, self.grpo_config.resolution))
                pil.save(os.path.join(tmpdir, f"{idx}.jpg"))

            sampled_prompts = [prompts[i] for i in sample_indices]

            # Safe reward retrieval
            sampled_rewards = []
            if "avg" in rewards:
                avg_rewards = rewards["avg"]
                # Handle scalar tensor
                if isinstance(avg_rewards, torch.Tensor) and avg_rewards.dim() == 0:
                    avg_rewards = avg_rewards.unsqueeze(0)

                for i in sample_indices:
                    if i < len(avg_rewards):
                        val = avg_rewards[i]
                        if isinstance(val, torch.Tensor):
                            val = val.item()
                        sampled_rewards.append(val)
                    else:
                        sampled_rewards.append(0.0)
            else:
                sampled_rewards = [0.0] * len(sample_indices)

            self.tracking.log(
                {
                    "images": [
                        {
                            "image": os.path.join(tmpdir, f"{idx}.jpg"),
                            "caption": f"{prompt:.100} | avg: {avg_reward:.2f}",
                        }
                        for idx, (prompt, avg_reward) in enumerate(zip(sampled_prompts, sampled_rewards))
                    ],
                },
                step=self.global_step,
            )

    def load_checkpoints(self, output_path: str, step: int):
        """
        Load checkpoint with safe handling of total_tokens.

        For GRPO training, total_tokens may not be tracked the same way as standard training,
        so we provide a default value if it's missing from the checkpoint.
        """
        # Ensure total_tokens is initialized before loading
        if not hasattr(self, "total_tokens"):
            self.total_tokens = 0

        # Call parent's load_checkpoints, but handle missing total_tokens gracefully
        try:
            super().load_checkpoints(output_path, step)
        except (KeyError, AttributeError) as e:
            if "total_tokens" in str(e):
                # If total_tokens is missing, set it to 0 and continue loading other state
                logger.warning(f"total_tokens not found in checkpoint, setting to 0")
                self.total_tokens = 0
                # Try to load the rest of the checkpoint manually
                rank = dist.get_rank()
                world_size = dist.get_world_size()
                extra_state_path = os.path.join(
                    output_path,
                    "extra_state",
                    f"extra_state_world_size_{world_size}_rank_{rank}.pt",
                )
                if os.path.exists(extra_state_path):
                    extra_state = torch.load(extra_state_path, weights_only=False)
                    # Load other state except total_tokens
                    if "rng" in extra_state:
                        self.load_rng_state(extra_state["rng"])
                    if "lr_scheduler_state" in extra_state:
                        self.scheduler.load_state_dict(extra_state["lr_scheduler_state"])
            else:
                raise
