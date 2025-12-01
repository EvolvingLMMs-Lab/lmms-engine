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
import torchvision.transforms.functional as TF
from accelerate.utils import send_to_device
from loguru import logger
from PIL import Image
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

        Unlike standard FSDP2 training which wraps the entire model,
        for GRPO we only wrap the language_model (transformer) to allow
        the inferencer to work correctly during sampling.

        This matches the approach in train_bagel.py where only the transformer
        is wrapped with FSDP, not the entire Bagel model.
        """
        import gc

        from torch.distributed.fsdp import MixedPrecisionPolicy

        from lmms_engine.utils.fsdp2_utils import (
            apply_fsdp2,
            fsdp2_load_full_state_dict,
        )

        # Setup reference model BEFORE FSDP wrapping if beta > 0
        if (
            hasattr(self.grpo_config, "train")
            and hasattr(self.grpo_config.train, "beta")
            and self.grpo_config.train.beta > 0
        ):
            logger.info("Setting up reference model for KL penalty")
            if hasattr(self.model, "language_model"):
                # Clone the language model for reference
                ref_model = type(self.model.language_model)(self.model.language_model.config)
                ref_model.load_state_dict(self.model.language_model.state_dict())
                ref_model.eval()
                ref_model.requires_grad_(False)
                self.language_model_ref = ref_model
                self.model.language_model_ref = ref_model

        # Only wrap language_model with FSDP2, not the entire model
        # This allows the inferencer to use the model correctly during sampling
        if hasattr(self.model, "language_model"):
            transformer = self.model.language_model
            transformer.config.use_cache = False

            # Get target device
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            device = torch.device(f"cuda:{local_rank}")

            # IMPORTANT: Move transformer to GPU BEFORE FSDP2 wrapping
            # This ensures all buffers (like rotary_emb.inv_freq) are on the correct device
            logger.info(f"Moving language_model to {device} before FSDP2 wrapping")
            transformer = transformer.to(device)

            # Enable gradient checkpointing if configured
            if self.args.gradient_checkpointing:
                if hasattr(transformer, "gradient_checkpointing_enable"):
                    transformer.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

            # Setup mixed precision policy
            if self.args.bf16:
                param_dtype = torch.bfloat16
            else:
                param_dtype = torch.float16

            reduce_dtype = getattr(torch, self.args.reduce_dtype)
            output_dtype = getattr(torch, self.args.output_dtype)
            mp_policy = MixedPrecisionPolicy(
                param_dtype=param_dtype,
                reduce_dtype=reduce_dtype,
                output_dtype=output_dtype,
            )

            fsdp_kwargs = {
                "reshard_after_forward": getattr(self.args, "fsdp_config", {}).get("reshard_after_forward", True),
                "mp_policy": mp_policy,
            }

            # Get transformer layer class to wrap
            transformer_cls_names_to_wrap = self.args.fsdp_config.get("transformer_layer_cls_to_wrap", None)

            # Save state dict before FSDP wrapping
            full_state = transformer.state_dict()

            logger.info(f"Applying FSDP2 to language_model (transformer only)")
            apply_fsdp2(transformer, fsdp_kwargs, transformer_cls_names_to_wrap)

            logger.info(f"Loading full state dict to language_model")
            fsdp2_load_full_state_dict(transformer, full_state)

            # CRITICAL: After FSDP2 wrapping and state loading, some buffers may be on CPU
            # We need to explicitly move them to the correct device
            # This is especially important for rotary_emb.inv_freq which is registered with persistent=False
            def move_buffers_to_device(module, target_device):
                for name, buf in module.named_buffers(recurse=False):
                    if buf is not None and buf.device != target_device:
                        logger.info(f"Moving buffer {name} from {buf.device} to {target_device}")
                        # Use register_buffer to properly update the buffer
                        module.register_buffer(name, buf.to(target_device), persistent=False)
                for child_name, child in module.named_children():
                    move_buffers_to_device(child, target_device)

            logger.info(f"Moving all buffers to {device}")
            move_buffers_to_device(transformer, device)

            # Put the wrapped transformer back into the model
            self.model.language_model = transformer

            logger.info(f"FSDP2 applied to language_model")

            del full_state
            gc.collect()
            torch.cuda.empty_cache()

            # Move all other components of Bagel to device (vae2llm, time_embedder, etc.)
            # language_model is already handled by FSDP
            for name, module in self.model.named_children():
                if name != "language_model":
                    module.to(device)

            # Move reference model to device if exists
            if self.language_model_ref is not None:
                self.language_model_ref = self.language_model_ref.to(device)
                self.model.language_model_ref = self.language_model_ref
        else:
            # Fallback to standard FSDP2 wrapping
            logger.warning("Model does not have language_model attribute, using standard FSDP2 wrapping")
            super().prepare_model()

        # Set fsdp2_model to the model for compatibility
        self.fsdp2_model = self.model

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

                    # Convert PIL Image to Tensor (C, H, W) in [0, 1]
                    # This matches the format expected by train_bagel.py logic and log_sample_images
                    img_tensor = TF.to_tensor(output_dict["image"])
                    images.append(img_tensor)
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
                    f"reward_{key}": value.mean()
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
                    self.tracking.log(
                        {
                            "group_size": group_size,
                            "trained_prompt_num": trained_prompt_num,
                            "zero_std_ratio": zero_std_ratio,
                            "reward_std_mean": reward_std_mean,
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

                    if dist.get_rank() == 0 and hasattr(self, "tracking"):
                        self.tracking.log(info_aggregated, step=self.global_step)

                    self.global_step += 1
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

        # Resume from checkpoint
        if resume_from_checkpoint:
            checkpoints = [f for f in os.listdir(self.args.output_dir) if f.startswith("checkpoint")]
            checkpoints.sort(key=lambda x: int(x.split("-")[1]))
            latest_checkpoint = checkpoints[-1]
            self.load_checkpoints(
                os.path.join(self.args.output_dir, latest_checkpoint),
                int(latest_checkpoint.split("-")[1]),
            )
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

            # Training phase
            self._training_phase(samples, advantages, tokenizer, epoch)

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
        """Evaluation phase (placeholder, can be implemented similarly to sampling)."""
        logger.info(f"Evaluation at epoch {epoch} (not implemented)")
        # TODO: Implement evaluation similar to train_bagel.py eval function

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
