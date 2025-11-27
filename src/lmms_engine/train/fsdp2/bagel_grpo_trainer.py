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
        model: nn.Module,
        args: TrainingArguments,
        train_dataset: DatasetType,
        eval_dataset: DatasetType = None,
        processing_class=None,
        data_collator=None,
        inferencer=None,
        reward_fn=None,
        eval_reward_fn=None,
        grpo_config=None,
        **kwargs,
    ) -> None:
        super().__init__(model, args, train_dataset, eval_dataset, processing_class, data_collator)

        # Get grpo_config from kwargs if not provided directly
        if grpo_config is None and "grpo_config" in kwargs:
            grpo_config = kwargs["grpo_config"]

        # Convert dict config to object if needed
        if grpo_config is not None and isinstance(grpo_config, dict):
            self.grpo_config = self._dict_to_config(grpo_config)
        else:
            self.grpo_config = grpo_config

        # Validate grpo_config
        if self.grpo_config is None:
            raise ValueError(
                "grpo_config must be provided for BagelGRPOTrainer (via grpo_config parameter or extra_kwargs)"
            )

        # GRPO-specific components - can be passed directly or initialized from config
        self.inferencer = inferencer
        self.reward_fn = reward_fn
        self.eval_reward_fn = eval_reward_fn

        # If not provided, try to initialize from kwargs or model attributes
        if self.inferencer is None or self.reward_fn is None:
            self._initialize_grpo_components(kwargs)

        # Executor for async reward computation
        self.executor = futures.ThreadPoolExecutor(max_workers=8)

        # Stat tracker for per-prompt normalization
        if hasattr(grpo_config, "per_prompt_stat_tracking") and grpo_config.per_prompt_stat_tracking:
            from lmms_engine.utils.tracking import PerPromptStatTracker

            self.stat_tracker = PerPromptStatTracker(grpo_config.sample.global_std)
        else:
            self.stat_tracker = None

        # Reference model for KL penalty (if beta > 0)
        self.language_model_ref = None
        if hasattr(grpo_config, "train") and hasattr(grpo_config.train, "beta") and grpo_config.train.beta > 0:
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
            # Fallback to flow_grpo paths
            from lmms_engine.models.bagel.data_utils import add_special_tokens
            from lmms_engine.models.bagel.inferencer import InterleaveInferencer
            from lmms_engine.models.bagel.transforms import ImageTransform

        # Get tokenizer
        tokenizer = self.processing_class
        if hasattr(tokenizer, "tokenizer"):
            tokenizer = tokenizer.tokenizer

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
        """Prepare model and optionally reference model for KL penalty."""
        super().prepare_model()

        # Setup reference model if beta > 0
        if (
            hasattr(self.grpo_config, "train")
            and hasattr(self.grpo_config.train, "beta")
            and self.grpo_config.train.beta > 0
        ):
            logger.info("Setting up reference model for KL penalty")
            # Create reference model from current model
            # Note: This should be done before FSDP wrapping
            if hasattr(self.model, "language_model"):
                ref_model = type(self.model.language_model)(self.model.language_model.config)
                ref_model.load_state_dict(self.model.language_model.state_dict())
                ref_model.eval()
                ref_model.requires_grad_(False)
                # Move to same device as main model
                device = next(self.fsdp2_model.parameters()).device
                ref_model = ref_model.to(device)
                self.language_model_ref = ref_model
                # Store in model for access during training
                if hasattr(self.model, "language_model_ref"):
                    self.model.language_model_ref = ref_model

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
            if hasattr(self.processing_class, "tokenizer"):
                tokenizer = self.processing_class.tokenizer
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

            # Compute rewards asynchronously
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
        for sample in tqdm(
            samples,
            desc="Waiting for rewards",
            disable=dist.get_rank() != 0,
            position=0,
        ):
            rewards, reward_metadata = sample["rewards"].result()
            sample["rewards"] = {
                key: torch.as_tensor(value, device=self.fsdp2_model.device).float() for key, value in rewards.items()
            }

        # Collate samples
        samples_collated = {
            k: torch.cat([s[k] for s in samples], dim=0)
            if not isinstance(samples[0][k], dict)
            else {sub_key: torch.cat([s[k][sub_key] for s in samples], dim=0) for sub_key in samples[0][k]}
            for k in samples[0].keys()
        }

        return samples_collated, images, prompts, rewards

    def _compute_advantages(self, samples: Dict, prompts: List[str], tokenizer):
        """
        Compute advantages from rewards.

        Returns:
            advantages: Tensor of advantages
        """
        # Gather rewards across processes
        gathered_rewards = {}
        for key, value in samples["rewards"].items():
            gathered_value = self._gather_tensor(value)
            gathered_rewards[key] = gathered_value.cpu().numpy()

        # Log rewards
        if dist.get_rank() == 0:
            reward_metrics = {
                f"reward_{key}": value.mean()
                for key, value in gathered_rewards.items()
                if "_strict_accuracy" not in key and "_accuracy" not in key
            }
            if hasattr(self, "tracking"):
                self.tracking.log(reward_metrics, step=self.global_step)

        # Compute advantages
        if self.stat_tracker is not None:
            # Per-prompt normalization
            prompt_ids = self._gather_tensor(samples["prompt_ids"]).cpu().numpy()
            prompts_decoded = tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
            advantages = self.stat_tracker.update(prompts_decoded, gathered_rewards["avg"])

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
            advantages = (gathered_rewards["avg"] - gathered_rewards["avg"].mean()) / (
                gathered_rewards["avg"].std() + 1e-4
            )

        # Ungather advantages to keep only entries for this process
        advantages = torch.as_tensor(advantages)
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        advantages = advantages.reshape(world_size, -1, advantages.shape[-1])[rank]
        advantages = advantages.to(self.fsdp2_model.device)

        return advantages

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

        # Set training mode but disable dropout in some layers
        if hasattr(self.fsdp2_model, "module"):
            self.fsdp2_model.module.training = False
            if hasattr(self.fsdp2_model.module, "model"):
                self.fsdp2_model.module.model.training = False

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
                    cur_sample = {k: v[j] for k, v in sample.items()}

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
                            transformer=self.fsdp2_model,
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
        if hasattr(self.processing_class, "tokenizer"):
            tokenizer = self.processing_class.tokenizer
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
            samples, images, prompts, rewards = self._sampling_phase(epoch, train_iter)

            # Log sample images periodically
            if epoch % 5 == 0 and rank == 0:
                self._log_sample_images(images, prompts, rewards, epoch)

            # Compute advantages
            advantages = self._compute_advantages(samples, prompts, tokenizer)
            samples["advantages"] = advantages
            del samples["rewards"]  # Free memory

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
        sample_indices = random.sample(range(len(images)), num_samples)

        with tempfile.TemporaryDirectory() as tmpdir:
            for idx, i in enumerate(sample_indices):
                image = images[i].cpu().numpy()
                pil = Image.fromarray((image.transpose(1, 2, 0) * 255).astype(np.uint8))
                pil = pil.resize((self.grpo_config.resolution, self.grpo_config.resolution))
                pil.save(os.path.join(tmpdir, f"{idx}.jpg"))

            sampled_prompts = [prompts[i] for i in sample_indices]
            sampled_rewards = [rewards["avg"][i] for i in sample_indices]

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
