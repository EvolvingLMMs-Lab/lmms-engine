import hashlib
import os
import random
import tempfile
import time
from collections import defaultdict
from concurrent import futures
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from PIL import Image
from torch.distributed.fsdp import register_fsdp_forward_method
from torch.utils.data import Dataset, IterableDataset
from tqdm import tqdm

from lmms_engine.train.config import TrainingArguments
from lmms_engine.train.fsdp2.fsdp2_trainer import FSDP2SFTTrainer
from lmms_engine.train.registry import TRAINER_REGISTER
from lmms_engine.utils import TrainUtilities
from lmms_engine.utils.tracking import Tracking

DatasetType = Union[Dataset, IterableDataset]


def _unwrap_module(m: nn.Module) -> nn.Module:
    # Works for FSDP and common wrappers.
    return getattr(m, "module", m)


@contextmanager
def _temporary_set_train_mode(module: nn.Module, train: bool = True):
    """Temporarily force training flags for module (and all submodules).

    Bagel's Qwen2 stack dispatches between forward_train and forward_inference based on
    `self.training`, so we need a safe way to ensure forward_train is used when we want
    token-level logprob gradients (e.g., think RL).
    """
    mods = list(module.modules())
    prev = [m.training for m in mods]
    try:
        for m in mods:
            m.training = train
        yield
    finally:
        for m, p in zip(mods, prev):
            m.training = p


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
        super().__init__(model, args, train_dataset, eval_dataset, processing_class, data_collator)

        grpo_config = kwargs.pop("grpo_config", None)

        if grpo_config is None and hasattr(train_dataset, "config"):
            dataset_extra_kwargs = getattr(train_dataset.config, "extra_kwargs", None)
            if dataset_extra_kwargs and isinstance(dataset_extra_kwargs, dict):
                grpo_config = dataset_extra_kwargs.get("grpo_config", None)

        if grpo_config is not None and isinstance(grpo_config, dict):
            self.grpo_config = self._dict_to_config(grpo_config)
        else:
            self.grpo_config = grpo_config

        self.inferencer = kwargs.pop("inferencer", None)
        self.reward_fn = kwargs.pop("reward_fn", None)
        self.eval_reward_fn = kwargs.pop("eval_reward_fn", None)

        if self.inferencer is None or self.reward_fn is None:
            self._initialize_grpo_components(kwargs)

        # Executor for async reward computation
        self.executor = futures.ThreadPoolExecutor(max_workers=8)

        if hasattr(self.grpo_config, "per_prompt_stat_tracking") and self.grpo_config.per_prompt_stat_tracking:
            from lmms_engine.utils.tracking import PerPromptStatTracker

            self.stat_tracker = PerPromptStatTracker(self.grpo_config.sample.global_std)
        else:
            self.stat_tracker = None

        self.language_model_ref = None
        if (
            hasattr(self.grpo_config, "train")
            and hasattr(self.grpo_config.train, "beta")
            and self.grpo_config.train.beta > 0
        ):
            pass

    def _initialize_grpo_components(self, kwargs):
        import torch.distributed as dist

        device = f"cuda:{dist.get_rank()}" if dist.is_initialized() else "cuda:0"

        if self.inferencer is None:
            if "inferencer" in kwargs:
                self.inferencer = kwargs["inferencer"]
            else:
                self.inferencer = self._create_inferencer_from_model()

        # Initialize reward functions if not provided
        if self.reward_fn is None:
            if "reward_fn" in kwargs:
                self.reward_fn = kwargs["reward_fn"]
            else:
                reward_fn_config = getattr(self.grpo_config, "reward_fn", {})

                from types import SimpleNamespace

                if isinstance(reward_fn_config, SimpleNamespace):
                    reward_fn_config = {k: v for k, v in reward_fn_config.__dict__.items()}
                    logger.info(f"Converted SimpleNamespace reward_fn_config to dict: {reward_fn_config}")

                if isinstance(reward_fn_config, str):
                    reward_fn_config = {reward_fn_config: 1.0}
                elif not isinstance(reward_fn_config, dict):
                    reward_fn_config = {}

                import lmms_engine.utils.rewards.rewards

                self.reward_fn = lmms_engine.utils.rewards.rewards.multi_score(device, reward_fn_config)
                logger.info(f"Initialized reward function with config: {reward_fn_config}")

        if self.eval_reward_fn is None:
            if "eval_reward_fn" in kwargs:
                self.eval_reward_fn = kwargs["eval_reward_fn"]
            else:
                self.eval_reward_fn = self.reward_fn  # Use same function for eval

    def _create_inferencer_from_model(self):
        from lmms_engine.models.bagel.inferencer import InterleaveInferencer

        tokenizer = self.processing_class.processor
        new_token_ids = self.processing_class.new_token_ids

        if hasattr(self.model, "vae_model"):
            vae_model = self.model.vae_model
        else:
            logger.warning("Could not find vae_model in model. Inferencer may not work correctly.")
            vae_model = None

        # Create transforms
        vae_transform = self.processing_class.vae_image_transform
        vit_transform = self.processing_class.vit_image_transform

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

    # =========================
    # Small internal helpers
    # (keep behavior identical; reduce duplication)
    # =========================
    def _get_tokenizer(self):
        """BagelDataProcessor stores tokenizer in `.processor` (not `.tokenizer`)."""
        return getattr(self.processing_class, "processor", self.processing_class)

    def _parse_rl_prompt_batch(self, batch):
        """
        Normalize dataloader outputs to a unified format:
          - Text-only: (prompts, metadatas)
          - Image edit: (prompts, metadatas, images, prompt_with_image_paths)
        Returns: (prompts, prompt_metadata, input_images)
        """
        if isinstance(batch, (list, tuple)):
            if len(batch) == 2:
                prompts, prompt_metadata = batch
                input_images = None
            elif len(batch) >= 4:
                prompts, prompt_metadata, input_images, _ = batch[:4]
            else:
                logger.warning(f"Unexpected batch format with {len(batch)} elements, treating as text-only")
                prompts = batch[0]
                prompt_metadata = batch[1] if len(batch) > 1 else [{}] * len(prompts)
                input_images = None
        elif isinstance(batch, dict):
            # Some callers may pass dict batches; keep a permissive fallback.
            prompts = batch.get("prompt", batch.get("text", []))
            prompt_metadata = batch.get("metadata", [{}] * len(prompts))
            input_images = batch.get("image", None)
        else:
            logger.warning(f"Unexpected batch type: {type(batch)}, treating as text-only")
            prompts = batch
            prompt_metadata = [{}] * len(prompts) if hasattr(prompts, "__len__") else [{}]
            input_images = None
        return prompts, prompt_metadata, input_images

    def _tokenize_prompts(self, prompts, tokenizer):
        return tokenizer(
            prompts,
            padding="max_length",
            max_length=256,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.fsdp2_model.device)

    def _normalize_rewards_avg_list(self, rewards: Dict, batch_size: int) -> List[float]:
        """
        Normalize rewards['avg'] to a python list[float] of length == batch_size.
        Matches existing behavior (including best-effort mismatch fixes).
        """
        if "avg" not in rewards:
            logger.warning(f"No 'avg' in rewards. Keys: {rewards.keys()}")
            return [0.0] * batch_size

        avg_val = rewards["avg"]
        if isinstance(avg_val, torch.Tensor):
            avg_len = avg_val.shape[0] if avg_val.dim() > 0 else 1
            avg_list = avg_val.cpu().tolist() if avg_val.dim() > 0 else [avg_val.item()]
        elif isinstance(avg_val, (list, tuple)):
            avg_len = len(avg_val)
            avg_list = list(avg_val)
        else:
            avg_len = 1
            avg_list = [float(avg_val)]

        if avg_len != batch_size:
            logger.error(
                f"Reward length mismatch! rewards['avg'] len: {avg_len}, batch_size: {batch_size}, "
                f"rewards['avg'] type: {type(rewards['avg'])}, value: {rewards['avg']}"
            )
            if avg_len == 0:
                logger.warning("Empty rewards received. Filling with zeros.")
                return [0.0] * batch_size
            if avg_len == 1 and batch_size > 1:
                logger.warning(f"Single reward for batch of {batch_size}. Repeating reward.")
                return avg_list * batch_size
            logger.warning("Cannot fix mismatch. Using zeros.")
            return [0.0] * batch_size

        return [float(x) for x in avg_list]

    def _rewards_to_tensor_dict(self, rewards: Dict) -> Dict[str, torch.Tensor]:
        """Convert reward values to float tensors on the training device."""
        out: Dict[str, torch.Tensor] = {}
        device = self.fsdp2_model.device
        for key, value in rewards.items():
            if isinstance(value, torch.Tensor):
                out[key] = value.to(device=device).float()
            elif isinstance(value, (list, tuple)):
                out[key] = torch.as_tensor(value, device=device).float()
            else:
                out[key] = torch.tensor([float(value)], device=device).float()
        return out

    def prepare_model(self):
        """
        Prepare model for GRPO training.
        """
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
        register_fsdp_forward_method(self.model, "forward_cache_update_vae")
        register_fsdp_forward_method(self.model, "forward_cache_update_vit")
        register_fsdp_forward_method(self.model, "forward_cache_update_text")
        # NOTE:
        # Think mode calls InterleaveInferencer.gen_text() -> Bagel.generate_text().
        # Without registering generate_text, the call may bypass FSDP's forward-method wrapper
        # and leave newly-created token tensors on CPU, causing:
        #   RuntimeError: Expected all tensors to be on the same device ... index is on cpu ...
        # Registering it makes device/DTensor handling consistent with other registered methods.
        register_fsdp_forward_method(self.model, "generate_text")
        register_fsdp_forward_method(self.model, "generate_image")
        register_fsdp_forward_method(self.model, "_forward_flow")
        register_fsdp_forward_method(self.model.vae_model, "decode")
        register_fsdp_forward_method(self.model.vae_model, "encode")

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

        Supports both:
        - Text-only generation: dataloader returns (prompts, metadatas)
        - Image editing: dataloader returns (prompts, metadatas, images, prompt_with_image_paths)

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
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_dataloader)
                batch = next(train_iter)

            prompts, prompt_metadata, input_images = self._parse_rl_prompt_batch(batch)

            # Log whether this is an image edit task
            is_image_edit = input_images is not None
            if i == 0 and dist.get_rank() == 0:
                logger.info(f"[DEBUG] Task type: {'Image Edit' if is_image_edit else 'Text-to-Image Generation'}")

            tokenizer = self._get_tokenizer()
            prompt_ids = self._tokenize_prompts(prompts, tokenizer)

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
            think_texts: List[Optional[str]] = []

            # Use autocast if configured
            autocast_dtype = torch.bfloat16 if self.args.bf16 else torch.float16
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                for idx, prompt in enumerate(prompts):
                    if self.grpo_config.sample.same_latent:
                        generator = generators[idx : idx + 1]
                    else:
                        generator = None

                    # Get input image for this sample (if image edit task)
                    input_image = input_images[idx] if input_images is not None else None

                    with torch.no_grad():
                        # IMPORTANT:
                        # InterleaveInferencer.__call__ returns output_list[0], which becomes a string when think=True.
                        # To support think mode, call interleave_inference() directly and unpack outputs.
                        think_kwargs = self._get_think_kwargs()
                        inference_hyper = self._get_inference_hyperparams()

                        input_list = []
                        if input_image is not None:
                            input_list.append(input_image)
                        input_list.append(prompt)

                        if think_kwargs.get("think", False):
                            output_list = self.inferencer.interleave_inference(
                                input_list,
                                think=True,
                                understanding_output=False,
                                max_think_token_n=int(think_kwargs.get("max_think_token_n", 1000)),
                                do_sample=bool(think_kwargs.get("do_sample", True)),
                                text_temperature=float(think_kwargs.get("text_temperature", 0.3)),
                                cfg_text_scale=self.grpo_config.sample.guidance_scale,
                                num_timesteps=self.grpo_config.sample.num_steps,
                                noise_level=self.grpo_config.sample.noise_level,
                                grpo_config=self.grpo_config,
                                accelerator=None,
                                generators=generator,
                                **inference_hyper,
                            )
                            think_text = output_list[0]
                            output_dict = output_list[1]
                        else:
                            output_list = self.inferencer.interleave_inference(
                                input_list,
                                think=False,
                                understanding_output=False,
                                cfg_text_scale=self.grpo_config.sample.guidance_scale,
                                num_timesteps=self.grpo_config.sample.num_steps,
                                noise_level=self.grpo_config.sample.noise_level,
                                grpo_config=self.grpo_config,
                                accelerator=None,
                                generators=generator,
                                **inference_hyper,
                            )
                            think_text = None
                            output_dict = output_list[0]

                    # Image is already tensor (C, H, W) in [0, 1], same as flow_grpo
                    images.append(output_dict["image"])
                    latents.append(output_dict["all_latents"])
                    log_probs.append(output_dict["all_log_probs"])
                    timesteps.append(output_dict["timesteps"])
                    think_texts.append(think_text)

            # Stack tensors
            stacked_inner_latents = [torch.stack(inner_list, dim=0) for inner_list in latents]
            latents = torch.stack(stacked_inner_latents, dim=0)
            stacked_inner_log_probs = [torch.stack(inner_list, dim=0) for inner_list in log_probs]
            log_probs = torch.stack(stacked_inner_log_probs, dim=0)
            timesteps = torch.stack(timesteps, dim=0)
            images = torch.stack(images, dim=0)

            # For image editing: pass input_images as ref_images for image_similarity reward
            if input_images is not None:
                rewards = self.executor.submit(
                    self.reward_fn, images, prompts, prompt_metadata, ref_images=input_images, only_strict=True
                )
            else:
                rewards = self.executor.submit(self.reward_fn, images, prompts, prompt_metadata, only_strict=True)
            time.sleep(0)  # Yield to start reward computation

            # Build sample dict
            sample_dict = {
                "prompt_ids": prompt_ids,
                "timesteps": timesteps,
                "latents": latents[:, :-1],  # Each entry is the latent before timestep t
                "prev_latents": latents[:, 1:],  # Each entry is the latent after timestep t
                "log_probs": log_probs,
                "rewards": rewards,
            }

            # Save input images for image edit task (used in training phase)
            # Note: input_images is a list of PIL Images, we keep them as-is for training
            if input_images is not None:
                sample_dict["input_images"] = input_images

            # Save think texts (used to replay context during training + think RL)
            if any(t is not None for t in think_texts):
                sample_dict["think_texts"] = think_texts

            samples.append(sample_dict)

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

            rewards["avg"] = self._normalize_rewards_avg_list(rewards, batch_size)
            sample["rewards"] = self._rewards_to_tensor_dict(rewards)

        # Collate samples
        # Handle input_images specially (list of PIL Images, not tensors)
        samples_collated = {}
        for k in samples[0].keys():
            if k == "input_images":
                # Concatenate lists of PIL Images
                all_images = []
                for s in samples:
                    if "input_images" in s and s["input_images"] is not None:
                        all_images.extend(s["input_images"])
                samples_collated[k] = all_images if all_images else None
            elif k == "think_texts":
                # Flatten list-of-think-texts to align with concatenated tensors
                all_think = []
                for s in samples:
                    vals = s.get("think_texts", None)
                    if vals is not None:
                        all_think.extend(list(vals))
                samples_collated[k] = all_think if all_think else None
            elif isinstance(samples[0][k], dict):
                # Handle nested dicts (e.g., rewards)
                samples_collated[k] = {
                    sub_key: torch.cat([s[k][sub_key] for s in samples], dim=0) for sub_key in samples[0][k]
                }
            elif isinstance(samples[0][k], torch.Tensor):
                # Handle tensors
                samples_collated[k] = torch.cat([s[k] for s in samples], dim=0)
            else:
                # Handle other types (e.g., lists)
                samples_collated[k] = [s[k] for s in samples]

        # Get rewards for the last batch for logging
        # Note: samples is a list of dicts, each dict contains rewards for that batch
        last_batch_rewards = samples[-1]["rewards"]

        # Get last batch input_images for logging (for image edit tasks)
        last_batch_input_images = samples[-1].get("input_images", None)

        return samples_collated, images, prompts, last_batch_rewards, last_batch_input_images

    def _get_mixed_resolution_cfg(self) -> Dict[str, Any]:
        """
        Mixed-resolution / variable-shape GRPO mode.

        Motivation:
        - In edit tasks, Bagel's inferencer derives `image_shapes` from the *resized* source image size
          (see InterleaveInferencer.interleave_inference), which can vary with source aspect ratio.
        - The default sampling phase collates rollout tensors across multiple sampled batches using
          `torch.cat`, which requires identical latent spatial shapes and will crash when shapes differ.

        We provide a config-gated alternative path that keeps rollouts "ragged" (per-sample tensors),
        computes bucketed advantages per (H, W), and trains with a fixed step schedule so distributed
        training remains synchronized.

        Config (YAML):
          grpo_config:
            sample:
              mixed_resolution:
                enabled: true
                adv_norm: per_bucket   # per_bucket | global
                log_topk: 10
        """
        sample_cfg = getattr(self.grpo_config, "sample", None)
        mr = getattr(sample_cfg, "mixed_resolution", None) if sample_cfg is not None else None

        # Backward-compatible alias (bool): sample.bucket_by_image_size: true
        if mr is None and sample_cfg is not None:
            alias = getattr(sample_cfg, "bucket_by_image_size", None)
            if isinstance(alias, bool):
                return {"enabled": alias, "adv_norm": "per_bucket", "log_topk": 10}

        if mr is None:
            return {"enabled": False}

        if isinstance(mr, bool):
            return {"enabled": bool(mr), "adv_norm": "per_bucket", "log_topk": 10}

        # `mr` is typically a SimpleNamespace (from _dict_to_config), but accept dict for safety.
        if isinstance(mr, dict):
            enabled = bool(mr.get("enabled", False))
            adv_norm = str(mr.get("adv_norm", mr.get("advantage_norm", "per_bucket")))
            log_topk = int(mr.get("log_topk", 10))
            return {"enabled": enabled, "adv_norm": adv_norm, "log_topk": log_topk}

        enabled = bool(getattr(mr, "enabled", False))
        adv_norm = str(getattr(mr, "adv_norm", getattr(mr, "advantage_norm", "per_bucket")))
        log_topk = int(getattr(mr, "log_topk", 10))
        return {"enabled": enabled, "adv_norm": adv_norm, "log_topk": log_topk}

    def _sampling_phase_mixed(self, epoch: int, train_iter):
        """
        Sampling phase for mixed-resolution mode.

        Key differences from `_sampling_phase`:
        - Keep rollout tensors per-sample (no cross-batch `torch.cat`).
        - Compute rewards per-sample to avoid stacking images with different H/W.
        - Return a list of batch records, preserving the fixed step schedule:
            len(samples_batched) == num_batches_per_epoch
        """
        self.fsdp2_model.eval()
        samples_batched: List[Dict[str, Any]] = []

        num_batches_per_epoch = self.grpo_config.sample.num_batches_per_epoch

        tokenizer = self._get_tokenizer()

        # For logging only (we will resize to fixed resolution)
        last_batch_output_images: List[torch.Tensor] = []
        last_batch_prompts: List[str] = []
        last_batch_input_images: Optional[List] = None

        autocast_dtype = torch.bfloat16 if self.args.bf16 else torch.float16

        for i in tqdm(
            range(num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling(mixed_res)",
            disable=dist.get_rank() != 0,
            position=0,
        ):
            if hasattr(train_iter, "set_epoch"):
                train_iter.set_epoch(epoch * num_batches_per_epoch + i)

            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_dataloader)
                batch = next(train_iter)

            prompts, prompt_metadata, input_images = self._parse_rl_prompt_batch(batch)

            prompt_ids = self._tokenize_prompts(prompts, tokenizer)

            # Token accounting (same as fixed mode)
            self.total_tokens += self._calculate_tokens_for_batch(prompt_ids, num_images=1)

            generators = create_generators(prompts, base_seed=42)

            batch_rollouts: List[Dict[str, Any]] = []
            batch_think_texts: List[Optional[str]] = []

            # Mixed-res: generate and score per sample to avoid stacking ragged tensors.
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                for idx, prompt in enumerate(prompts):
                    generator = generators[idx : idx + 1] if self.grpo_config.sample.same_latent else None
                    input_image = input_images[idx] if input_images is not None else None
                    meta_i = (
                        prompt_metadata[idx] if isinstance(prompt_metadata, list) and idx < len(prompt_metadata) else {}
                    )

                    with torch.no_grad():
                        think_kwargs = self._get_think_kwargs()
                        inference_hyper = self._get_inference_hyperparams()

                        input_list = []
                        if input_image is not None:
                            input_list.append(input_image)
                        input_list.append(prompt)

                        if think_kwargs.get("think", False):
                            output_list = self.inferencer.interleave_inference(
                                input_list,
                                think=True,
                                understanding_output=False,
                                max_think_token_n=int(think_kwargs.get("max_think_token_n", 1000)),
                                do_sample=bool(think_kwargs.get("do_sample", True)),
                                text_temperature=float(think_kwargs.get("text_temperature", 0.3)),
                                cfg_text_scale=self.grpo_config.sample.guidance_scale,
                                num_timesteps=self.grpo_config.sample.num_steps,
                                noise_level=self.grpo_config.sample.noise_level,
                                grpo_config=self.grpo_config,
                                accelerator=None,
                                generators=generator,
                                **inference_hyper,
                            )
                            think_text = output_list[0]
                            output_dict = output_list[1]
                        else:
                            output_list = self.inferencer.interleave_inference(
                                input_list,
                                think=False,
                                understanding_output=False,
                                cfg_text_scale=self.grpo_config.sample.guidance_scale,
                                num_timesteps=self.grpo_config.sample.num_steps,
                                noise_level=self.grpo_config.sample.noise_level,
                                grpo_config=self.grpo_config,
                                accelerator=None,
                                generators=generator,
                                **inference_hyper,
                            )
                            think_text = None
                            output_dict = output_list[0]

                    img = output_dict["image"]  # (C,H,W) float in [0,1]
                    H, W = int(img.shape[-2]), int(img.shape[-1])
                    bucket_key: Tuple[int, int] = (H, W)

                    # Stack per-sample rollout tensors (shape is allowed to differ across samples)
                    all_latents = torch.stack(output_dict["all_latents"], dim=0)  # [T+1, ...]
                    all_log_probs = torch.stack(output_dict["all_log_probs"], dim=0)  # [T, ...]
                    timesteps = output_dict["timesteps"]
                    if not isinstance(timesteps, torch.Tensor):
                        timesteps = torch.as_tensor(timesteps, device=self.fsdp2_model.device)
                    else:
                        timesteps = timesteps.to(self.fsdp2_model.device)

                    # Schedule reward computation per sample (avoids ragged torch.stack)
                    img_batched = img.unsqueeze(0)
                    if input_image is not None:
                        rewards_future = self.executor.submit(
                            self.reward_fn,
                            img_batched,
                            [prompt],
                            [meta_i],
                            ref_images=[input_image],
                            only_strict=True,
                        )
                    else:
                        rewards_future = self.executor.submit(
                            self.reward_fn,
                            img_batched,
                            [prompt],
                            [meta_i],
                            only_strict=True,
                        )
                    time.sleep(0)

                    batch_rollouts.append(
                        {
                            "bucket_key": bucket_key,
                            "timesteps": timesteps,
                            "latents": all_latents[:-1],
                            "prev_latents": all_latents[1:],
                            "log_probs": all_log_probs,
                            "rewards_future": rewards_future,
                        }
                    )
                    batch_think_texts.append(think_text)

                    # Keep last batch images for logging only
                    if i == num_batches_per_epoch - 1:
                        last_batch_output_images.append(img)

            batch_record = {
                "prompt_ids": prompt_ids,
                "prompts": list(prompts),
                "think_texts": batch_think_texts if any(t is not None for t in batch_think_texts) else None,
                "input_images": input_images,
                "rollouts": batch_rollouts,
            }
            samples_batched.append(batch_record)

            if i == num_batches_per_epoch - 1:
                last_batch_prompts = list(prompts)
                last_batch_input_images = input_images

        # Resolve rewards futures and convert to tensors
        device = self.fsdp2_model.device
        for batch in tqdm(
            samples_batched,
            desc="Waiting for rewards(mixed_res)",
            disable=dist.get_rank() != 0,
            position=0,
        ):
            for rollout in batch["rollouts"]:
                try:
                    rewards, _ = rollout["rewards_future"].result()
                except Exception as e:
                    logger.error(f"Error getting reward result (mixed_res): {e}", exc_info=True)
                    rewards = {"avg": [0.0]}

                # Normalize to list form of len=1
                if "avg" not in rewards:
                    rewards["avg"] = [0.0]
                avg_val = rewards["avg"]
                if isinstance(avg_val, torch.Tensor):
                    avg_val = avg_val.detach().cpu().view(-1).tolist()
                elif isinstance(avg_val, (list, tuple)):
                    avg_val = list(avg_val)
                else:
                    avg_val = [float(avg_val)]
                if len(avg_val) != 1:
                    avg_val = [float(avg_val[0])] if len(avg_val) > 0 else [0.0]
                rewards["avg"] = avg_val

                rollout_rewards: Dict[str, torch.Tensor] = {}
                for k, v in rewards.items():
                    if isinstance(v, torch.Tensor):
                        t = v.to(device=device).float().view(-1)
                    elif isinstance(v, (list, tuple)):
                        t = torch.as_tensor(v, device=device).float().view(-1)
                    else:
                        t = torch.tensor([float(v)], device=device).float()
                    if t.numel() == 0:
                        t = torch.tensor([0.0], device=device).float()
                    rollout_rewards[k] = t
                rollout["rewards"] = rollout_rewards
                rollout.pop("rewards_future", None)

        # Prepare logging outputs (resize to fixed resolution so `_log_sample_images` can stack)
        resolution = int(getattr(self.grpo_config, "resolution", 512))
        resized_for_logging: List[torch.Tensor] = []
        for img in last_batch_output_images:
            img4 = img.unsqueeze(0)  # [1,C,H,W]
            img4 = F.interpolate(img4, size=(resolution, resolution), mode="bilinear", align_corners=False)
            resized_for_logging.append(img4[0])
        images_for_logging = torch.stack(resized_for_logging, dim=0) if resized_for_logging else torch.empty(0)

        # Last batch rewards for logging
        last_batch_rewards = {"avg": torch.zeros((len(last_batch_prompts),), device=device)}
        if samples_batched:
            last_batch = samples_batched[-1]
            if last_batch.get("rollouts", None):
                avg_list = [
                    r.get("rewards", {}).get("avg", torch.tensor([0.0], device=device)).view(-1)[0]
                    for r in last_batch["rollouts"]
                ]
                last_batch_rewards = {"avg": torch.stack(avg_list, dim=0)}

        return samples_batched, images_for_logging, last_batch_prompts, last_batch_rewards, last_batch_input_images

    def _compute_advantages_mixed(self, samples_batched: List[Dict[str, Any]]):
        """
        Compute advantages for mixed-resolution mode.

        By default we normalize per (H,W) bucket using *global* bucket statistics across ranks,
        computed via `dist.all_gather_object` on small dicts (count/sum/sumsq). This avoids
        variable-length tensor all-gather.
        """
        mixed_cfg = self._get_mixed_resolution_cfg()
        adv_norm = str(mixed_cfg.get("adv_norm", "per_bucket")).lower()
        log_topk = int(mixed_cfg.get("log_topk", 10))

        device = self.fsdp2_model.device
        adv_clip_max = float(getattr(getattr(self.grpo_config, "train", None), "adv_clip_max", 5.0))

        bucket_keys: List[str] = []
        rewards_list: List[torch.Tensor] = []
        for batch in samples_batched:
            for rollout in batch.get("rollouts", []):
                r = rollout.get("rewards", {}).get("avg", None)
                if r is None:
                    continue
                rewards_list.append(r.view(-1)[0].float())
                bk = rollout.get("bucket_key", None)
                if isinstance(bk, (tuple, list)) and len(bk) == 2:
                    bucket_keys.append(f"{int(bk[0])}x{int(bk[1])}")
                else:
                    bucket_keys.append(str(bk))

        if not rewards_list:
            return torch.zeros(0, device=device)

        rewards_t = torch.stack(rewards_list, dim=0).to(device=device, dtype=torch.float32)

        # Global normalization (matches fixed-path behavior, but on mixed samples)
        if adv_norm == "global":
            gathered = self._gather_tensor(rewards_t)
            mean = gathered.mean()
            std = gathered.std()
            if std < 1e-6:
                advantages = rewards_t - mean
            else:
                advantages = (rewards_t - mean) / (std + 1e-4)
            advantages = torch.clamp(advantages, -adv_clip_max, adv_clip_max)
            idx = 0
            for batch in samples_batched:
                for rollout in batch.get("rollouts", []):
                    rollout["advantages"] = advantages[idx : idx + 1]
                    rollout.pop("rewards", None)
                    idx += 1
            return advantages

        # Per-bucket normalization using global bucket moments
        local_stats: Dict[str, Dict[str, float]] = {}
        for k, r in zip(bucket_keys, rewards_t.detach().cpu().tolist()):
            s = local_stats.setdefault(k, {"count": 0.0, "sum": 0.0, "sumsq": 0.0})
            s["count"] += 1.0
            s["sum"] += float(r)
            s["sumsq"] += float(r) * float(r)

        if dist.is_initialized() and dist.get_world_size() > 1:
            gathered_stats: List[Dict[str, Dict[str, float]]] = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered_stats, local_stats)
            global_stats: Dict[str, Dict[str, float]] = {}
            for d in gathered_stats:
                if not d:
                    continue
                for k, s in d.items():
                    g = global_stats.setdefault(k, {"count": 0.0, "sum": 0.0, "sumsq": 0.0})
                    g["count"] += float(s.get("count", 0.0))
                    g["sum"] += float(s.get("sum", 0.0))
                    g["sumsq"] += float(s.get("sumsq", 0.0))
        else:
            global_stats = local_stats

        # Global fallback moments (avoid all-zero advantages when a bucket has only 1 sample)
        global_count = max(1.0, sum(v["count"] for v in global_stats.values()))
        global_sum = sum(v["sum"] for v in global_stats.values())
        global_sumsq = sum(v["sumsq"] for v in global_stats.values())
        global_mean = global_sum / global_count
        global_var = max(global_sumsq / global_count - global_mean * global_mean, 0.0)
        global_std = global_var**0.5

        bucket_moments: Dict[str, Tuple[int, float, float]] = {}
        for k, s in global_stats.items():
            c = max(1.0, float(s["count"]))
            m = float(s["sum"]) / c
            var = max(float(s["sumsq"]) / c - m * m, 0.0)
            bucket_moments[k] = (int(c), m, var**0.5)

        eps = 1e-6
        adv_vals: List[float] = []
        for k, r in zip(bucket_keys, rewards_t.detach().cpu().tolist()):
            c, m, s = bucket_moments.get(k, (0, global_mean, global_std))
            use_global = c < 2 or s < eps
            mean = global_mean if use_global else m
            std = global_std if use_global else s
            if std < eps:
                adv = float(r - mean)
            else:
                adv = float((r - mean) / (std + 1e-4))
            adv_vals.append(adv)

        advantages = torch.tensor(adv_vals, device=device, dtype=torch.float32)
        advantages = torch.clamp(advantages, -adv_clip_max, adv_clip_max)

        # Attach back to rollouts (and drop rewards to save memory)
        idx = 0
        for batch in samples_batched:
            for rollout in batch.get("rollouts", []):
                rollout["advantages"] = advantages[idx : idx + 1]
                rollout.pop("rewards", None)
                idx += 1

        # Log per-bucket summary (top-k by count)
        if dist.get_rank() == 0 and hasattr(self, "tracking"):
            topk = sorted(bucket_moments.items(), key=lambda kv: kv[1][0], reverse=True)[: max(0, log_topk)]
            log_dict: Dict[str, Any] = {
                "mixed_res/num_buckets": len(bucket_moments),
                "mixed_res/global_reward_mean": float(global_mean),
                "mixed_res/global_reward_std": float(global_std),
            }
            for k, (c, m, s) in topk:
                log_dict[f"mixed_res/bucket/{k}/count"] = int(c)
                log_dict[f"mixed_res/bucket/{k}/reward_mean"] = float(m)
                log_dict[f"mixed_res/bucket/{k}/reward_std"] = float(s)
            self.tracking.log(log_dict, step=self.global_step)

        return advantages

    def _training_phase_mixed(self, samples_batched: List[Dict[str, Any]], tokenizer, epoch: int):
        """
        Training phase for mixed-resolution mode.

        Important constraints:
        - We keep the *same* outer step schedule: one `global_step` per sampled dataloader batch.
        - We always execute `bs` per-sample learn calls per batch, so all ranks stay synchronized.
        """
        self.fsdp2_model.train()

        # Ensure inferencer uses inference path inside Qwen2 components.
        if hasattr(self.model, "language_model"):
            self._force_inference_mode(self.model.language_model)

        num_inner_epochs = self.grpo_config.train.num_inner_epochs
        num_steps = int(self.grpo_config.sample.num_steps)
        timestep_shift = float(self.grpo_config.train.timestep_shift)
        device = self.fsdp2_model.device

        # Precompute the exact timestep grid used by generate_image_learn
        grid = torch.linspace(1, 0, num_steps, device=device, dtype=torch.float32)
        grid = timestep_shift * grid / (1 + (timestep_shift - 1) * grid)

        autocast_dtype = torch.bfloat16 if self.args.bf16 else torch.float16

        for inner_epoch in range(num_inner_epochs):
            for batch_idx, batch in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training(mixed_res)",
                disable=dist.get_rank() != 0,
                position=0,
            ):
                info = defaultdict(list)

                prompts: List[str] = list(batch.get("prompts", []))
                prompt_ids = batch.get("prompt_ids", None)
                input_images = batch.get("input_images", None)
                think_texts = batch.get("think_texts", None)
                rollouts = batch.get("rollouts", [])

                # Token accounting (same as fixed mode)
                if isinstance(prompt_ids, torch.Tensor):
                    self.total_tokens += self._calculate_tokens_for_batch(prompt_ids, num_images=1)

                # Think-RL (optional) once per batch
                if think_texts is not None and rollouts:
                    adv_vec = torch.cat(
                        [r.get("advantages", torch.zeros(1, device=device)) for r in rollouts], dim=0
                    ).view(-1)
                    try:
                        self._maybe_reinforce_think(prompts=prompts, think_texts=think_texts, advantages=adv_vec)
                    except Exception as e:
                        logger.warning(f"[think_rl] failed to update think policy (mixed_res): {e}", exc_info=True)

                inference_hyper = self._get_inference_hyperparams()
                think_kwargs = self._get_think_kwargs()

                def _to_cpu(x):
                    if isinstance(x, torch.Tensor):
                        return x.detach().cpu()
                    return x

                bs = len(rollouts)
                for j in tqdm(
                    range(bs),
                    desc="Batch Size",
                    position=1,
                    leave=False,
                    disable=dist.get_rank() != 0,
                ):
                    r = rollouts[j]
                    t = r["timesteps"].to(device=device, dtype=torch.float32)
                    indices = torch.argmin(torch.abs(t.unsqueeze(-1) - grid.unsqueeze(0)), dim=-1)
                    snapped_t = grid[indices]

                    cur_sample = {
                        "timesteps": snapped_t,
                        "latents": r["latents"].to(device),
                        "prev_latents": r["prev_latents"].to(device),
                        "log_probs": r["log_probs"].to(device),
                        "advantages": r.get("advantages", torch.zeros(1, device=device)).to(device),
                    }

                    input_image = input_images[j] if input_images is not None and j < len(input_images) else None

                    input_list = []
                    if (
                        think_kwargs.get("think", False)
                        and think_texts is not None
                        and j < len(think_texts)
                        and think_texts[j] is not None
                    ):
                        input_list.append(self._get_think_system_prompt())
                    if input_image is not None:
                        input_list.append(input_image)
                    if j < len(prompts):
                        input_list.append(prompts[j])
                    if (
                        think_kwargs.get("think", False)
                        and think_texts is not None
                        and j < len(think_texts)
                        and think_texts[j] is not None
                    ):
                        input_list.append(str(think_texts[j]))

                    with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                        output_list = self.inferencer.interleave_inference(
                            input_list,
                            think=False,
                            understanding_output=False,
                            learn=True,
                            sample=cur_sample,
                            grpo_config=self.grpo_config,
                            accelerator=None,
                            optimizer=self.optimizer,
                            transformer=self.model.language_model,
                            num_timesteps=self.grpo_config.sample.num_steps,
                            cfg_text_scale=self.grpo_config.sample.guidance_scale,
                            noise_level=self.grpo_config.sample.noise_level,
                            **inference_hyper,
                        )
                        output_dict = output_list[0]

                    info["clipfrac"].append(_to_cpu(output_dict["clipfrac"]))
                    info["clipfrac_gt_one"].append(_to_cpu(output_dict["clipfrac_gt_one"]))
                    info["clipfrac_lt_one"].append(_to_cpu(output_dict["clipfrac_lt_one"]))
                    info["policy_loss"].append(_to_cpu(output_dict["policy_loss"]))
                    info["kl_loss"].append(_to_cpu(output_dict["kl_loss"]))
                    info["loss"].append(_to_cpu(output_dict["loss"]))

                # Aggregate metrics for this batch and log (one global_step per batch)
                if info:
                    info_aggregated = {k: torch.mean(torch.stack(v).to(device)) for k, v in info.items()}
                    for key in info_aggregated:
                        info_aggregated[key] = self._reduce_tensor(info_aggregated[key])

                    info_aggregated.update({"epoch": epoch, "inner_epoch": inner_epoch})

                    # Sync and log token counts
                    total_tokens_tensor = torch.tensor(self.total_tokens, device=device, dtype=torch.long)
                    dist.all_reduce(total_tokens_tensor, op=dist.ReduceOp.SUM)
                    self.total_tokens = total_tokens_tensor.item()
                    if dist.get_rank() == 0:
                        info_aggregated["train/total_tokens"] = TrainUtilities.format_tokens(self.total_tokens)

                        info_aggregated_for_logging: Dict[str, Any] = {}
                        for k, v in info_aggregated.items():
                            if isinstance(v, torch.Tensor) and v.numel() == 1:
                                info_aggregated_for_logging[k] = v.item()
                            elif isinstance(v, (int, float, np.integer, np.floating)):
                                info_aggregated_for_logging[k] = float(v)
                        if hasattr(self, "tracking"):
                            self.tracking.log(info_aggregated_for_logging, step=self.global_step)

                    self.global_step += 1

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

        # Check if this is an image edit task
        has_input_images = "input_images" in samples and samples["input_images"] is not None
        input_images_list = samples.get("input_images", None)
        has_think_texts = "think_texts" in samples and samples["think_texts"] is not None
        think_texts_list = samples.get("think_texts", None)

        for inner_epoch in range(num_inner_epochs):
            # Rebatch for training
            # Handle input_images separately (list of PIL Images, not tensors)
            rebatch_size = total_batch_size // num_batches_per_epoch
            samples_batched = {}
            for k, v in samples.items():
                if k == "input_images":
                    continue  # Handle separately below
                if k == "think_texts":
                    continue  # Handle separately below
                elif isinstance(v, torch.Tensor):
                    samples_batched[k] = v.reshape(-1, rebatch_size, *v.shape[1:])
                else:
                    # Skip non-tensor values
                    continue

            # Convert dict to list of dicts for easier iteration
            samples_batched = [dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())]

            # Add input_images to each batch if present
            if has_input_images and input_images_list:
                for batch_idx, batch in enumerate(samples_batched):
                    start_idx = batch_idx * rebatch_size
                    end_idx = start_idx + rebatch_size
                    batch["input_images"] = input_images_list[start_idx:end_idx]
            if has_think_texts and think_texts_list:
                for batch_idx, batch in enumerate(samples_batched):
                    start_idx = batch_idx * rebatch_size
                    end_idx = start_idx + rebatch_size
                    batch["think_texts"] = think_texts_list[start_idx:end_idx]

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

                # Get input images for this batch if present (for image edit task)
                batch_input_images = sample.get("input_images", None)
                batch_think_texts = sample.get("think_texts", None)

                # Optional: REINFORCE update for think tokens, using image reward advantages as signal.
                # We do it once per mini-batch to keep overhead bounded.
                if batch_think_texts is not None and "advantages" in sample:
                    try:
                        adv_vec = sample["advantages"]
                        if isinstance(adv_vec, torch.Tensor):
                            adv_vec = adv_vec.view(-1)
                        self._maybe_reinforce_think(
                            prompts=prompts,
                            think_texts=batch_think_texts,
                            advantages=adv_vec,
                        )
                    except Exception as e:
                        logger.warning(f"[think_rl] failed to update think policy: {e}", exc_info=True)

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
                        if k == "input_images":
                            continue  # Handle separately
                        if k == "think_texts":
                            continue  # Handle separately
                        elif k == "advantages":
                            # Keep advantages as tensor for broadcasting
                            # If v is [batch_size], we need to select v[j] but keep as [1] tensor
                            if v.dim() == 1:
                                cur_sample[k] = v[j : j + 1]  # Keep as [1] tensor
                            else:
                                cur_sample[k] = v[j]
                        elif isinstance(v, torch.Tensor):
                            cur_sample[k] = v[j]

                    # Get input image for this sample (for image edit task)
                    input_image = batch_input_images[j] if batch_input_images is not None else None

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
                        inference_hyper = self._get_inference_hyperparams()
                        think_kwargs = self._get_think_kwargs()

                        # Replay sampled think context deterministically during training.
                        input_list = []
                        if (
                            think_kwargs.get("think", False)
                            and batch_think_texts is not None
                            and batch_think_texts[j] is not None
                        ):
                            input_list.append(self._get_think_system_prompt())
                        if input_image is not None:
                            input_list.append(input_image)
                        input_list.append(prompts[j])
                        if (
                            think_kwargs.get("think", False)
                            and batch_think_texts is not None
                            and batch_think_texts[j] is not None
                        ):
                            input_list.append(str(batch_think_texts[j]))

                        output_list = self.inferencer.interleave_inference(
                            input_list,
                            think=False,  # Never sample think during training; we replay context instead.
                            understanding_output=False,
                            learn=True,
                            sample=cur_sample,
                            grpo_config=self.grpo_config,
                            accelerator=None,
                            optimizer=self.optimizer,
                            transformer=self.model.language_model,
                            num_timesteps=self.grpo_config.sample.num_steps,
                            cfg_text_scale=self.grpo_config.sample.guidance_scale,
                            noise_level=self.grpo_config.sample.noise_level,
                            **inference_hyper,
                        )
                        output_dict = output_list[0]

                    # Immediately move results to CPU and clear GPU tensors to prevent OOM
                    # These metrics are only for logging, so they don't need to stay on GPU
                    def _to_cpu(x):
                        if isinstance(x, torch.Tensor):
                            return x.detach().cpu()
                        return x

                    info["clipfrac"].append(_to_cpu(output_dict["clipfrac"]))
                    info["clipfrac_gt_one"].append(_to_cpu(output_dict["clipfrac_gt_one"]))
                    info["clipfrac_lt_one"].append(_to_cpu(output_dict["clipfrac_lt_one"]))
                    info["policy_loss"].append(_to_cpu(output_dict["policy_loss"]))
                    info["kl_loss"].append(_to_cpu(output_dict["kl_loss"]))
                    info["loss"].append(_to_cpu(output_dict["loss"]))

                # Aggregate metrics when gradient sync happens
                if self._should_sync_gradients():
                    # Stack tensors (they're on CPU now) and compute mean
                    info_aggregated = {
                        k: torch.mean(torch.stack(v).to(self.fsdp2_model.device)) for k, v in info.items()
                    }
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

    def _get_think_kwargs(self) -> Dict:
        """
        Get think mode kwargs for inferencer if enabled.

        Returns:
            Dict of think mode parameters to pass to inferencer, empty dict if disabled.
        """
        think_config = getattr(self.grpo_config, "think", None)
        if think_config is not None and getattr(think_config, "enabled", False):
            return {
                "think": True,
                "max_think_token_n": getattr(think_config, "max_think_tokens", 1000),
                "do_sample": getattr(think_config, "do_sample", True),
                "text_temperature": getattr(think_config, "temperature", 0.3),
            }
        return {}

    def _get_think_system_prompt(self) -> str:
        # Keep behavior consistent with InterleaveInferencer.
        from lmms_engine.models.bagel.inferencer import GEN_THINK_SYSTEM_PROMPT

        return GEN_THINK_SYSTEM_PROMPT

    def _maybe_reinforce_think(
        self,
        *,
        prompts: List[str],
        think_texts: List[Optional[str]],
        advantages: torch.Tensor,
    ) -> None:
        """REINFORCE-style update for think text.

        We treat sampled think text as an action and maximize its token log-probability
        weighted by the image reward advantage. This updates the model to produce
        better think plans that improve generation/editing rewards, without changing
        the model architecture.
        """
        think_kwargs = self._get_think_kwargs()
        if not think_kwargs.get("think", False):
            return

        # IMPORTANT: Do NOT change original training behavior by default.
        # Think-RL must be explicitly enabled in config.
        think_cfg = getattr(self.grpo_config, "think", None)
        rl_enabled = bool(getattr(think_cfg, "rl_enabled", False)) if think_cfg is not None else False
        loss_coef = float(getattr(think_cfg, "rl_loss_coef", 0.0)) if think_cfg is not None else 0.0
        if not rl_enabled or loss_coef <= 0:
            return

        if not think_texts or all(t is None or len(str(t).strip()) == 0 for t in think_texts):
            return

        max_tokens = int(think_kwargs.get("max_think_token_n", 256))
        adv_clip_max = float(getattr(getattr(self.grpo_config, "train", None), "adv_clip_max", 5.0))

        tokenizer = getattr(self.processing_class, "processor", self.processing_class)
        new_token_ids = getattr(self.processing_class, "new_token_ids", None)
        if (
            not isinstance(new_token_ids, dict)
            or "bos_token_id" not in new_token_ids
            or "eos_token_id" not in new_token_ids
        ):
            logger.warning("[think_rl] new_token_ids not found; skip think RL update")
            return
        bos = int(new_token_ids["bos_token_id"])
        eos = int(new_token_ids["eos_token_id"])

        lm_outer = getattr(self.model, "language_model", None)
        if lm_outer is None:
            logger.warning("[think_rl] model has no language_model; skip think RL update")
            return

        qwen_lm = _unwrap_module(lm_outer)
        if hasattr(qwen_lm, "base_model"):
            qwen_lm = qwen_lm.base_model
        if not hasattr(qwen_lm, "model") or not hasattr(qwen_lm, "lm_head"):
            logger.warning(f"[think_rl] unexpected language_model={type(qwen_lm)}; skip think RL update")
            return

        device = self.fsdp2_model.device
        system_prompt = self._get_think_system_prompt()
        sys_ids = tokenizer.encode(system_prompt)
        sys_seg = [bos] + sys_ids + [eos]

        from lmms_engine.models.bagel.data_utils import (
            prepare_attention_mask_per_sample,
        )

        losses = []
        with _temporary_set_train_mode(qwen_lm, train=True):
            for i, (prompt, think_text) in enumerate(zip(prompts, think_texts)):
                if think_text is None or len(str(think_text).strip()) == 0:
                    continue

                adv = advantages[i]
                if isinstance(adv, torch.Tensor):
                    adv = adv.float().view(-1)[0]
                else:
                    adv = torch.tensor(float(adv), device=device, dtype=torch.float32)
                adv = torch.clamp(adv, -adv_clip_max, adv_clip_max)

                prompt_ids = tokenizer.encode(prompt)
                prompt_seg = [bos] + prompt_ids + [eos]

                think_ids = tokenizer.encode(str(think_text))
                if max_tokens > 0:
                    think_ids = think_ids[:max_tokens]
                think_seg = [bos] + think_ids + [eos]

                full_ids = sys_seg + prompt_seg + think_seg
                if len(full_ids) < 4:
                    continue

                input_ids = torch.tensor(full_ids[:-1], device=device, dtype=torch.long)
                labels = torch.tensor(full_ids[1:], device=device, dtype=torch.long)
                seq_len = input_ids.shape[0]
                packed_position_ids = torch.arange(seq_len, device=device, dtype=torch.long)
                packed_sequence = qwen_lm.model.embed_tokens(input_ids)

                attn_mask = prepare_attention_mask_per_sample([seq_len], ["causal"])
                if isinstance(attn_mask, list):
                    attn_mask = attn_mask[0]
                attn_mask = attn_mask.to(device)

                packed_und_token_indexes = torch.arange(seq_len, device=device, dtype=torch.long)
                packed_gen_token_indexes = packed_und_token_indexes.new_ones(size=[0])

                hidden = qwen_lm.model.forward_train(
                    packed_sequence=packed_sequence,
                    sample_lens=[seq_len],
                    attention_mask=[attn_mask],
                    packed_position_ids=packed_position_ids,
                    packed_und_token_indexes=packed_und_token_indexes,
                    packed_gen_token_indexes=packed_gen_token_indexes,
                )
                logits = qwen_lm.lm_head(hidden).float()
                logprobs = F.log_softmax(logits, dim=-1)
                token_logp = logprobs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

                sys_len = len(sys_seg)
                prompt_len = len(prompt_seg)
                offset_think_bos = sys_len + prompt_len
                start = max(0, min(offset_think_bos, token_logp.numel() - 1))
                think_token_logp = token_logp[start:]
                if think_token_logp.numel() == 0:
                    continue

                losses.append(-adv * think_token_logp.mean())

            if not losses:
                return

            loss = loss_coef * torch.stack(losses).mean()
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"[think_rl] invalid loss={loss}; skip step")
                self.optimizer.zero_grad(set_to_none=True)
                return

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        # Restore inference path required by inferencer (Qwen2 modules dispatch on self.training)
        if hasattr(self.model, "language_model"):
            self._force_inference_mode(self.model.language_model)

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

        # Log think mode configuration
        think_config = getattr(self.grpo_config, "think", None)
        if rank == 0:
            if think_config is not None and getattr(think_config, "enabled", False):
                logger.info(
                    f"[CONFIG] Think mode ENABLED: "
                    f"max_think_tokens={getattr(think_config, 'max_think_tokens', 1000)}, "
                    f"do_sample={getattr(think_config, 'do_sample', True)}, "
                    f"temperature={getattr(think_config, 'temperature', 0.3)}"
                )
            else:
                logger.info("[CONFIG] Think mode DISABLED")

        # Log mixed-resolution configuration
        mixed_cfg = self._get_mixed_resolution_cfg()
        if rank == 0:
            if mixed_cfg.get("enabled", False):
                logger.info(
                    f"[CONFIG] Mixed-resolution ENABLED: adv_norm={mixed_cfg.get('adv_norm', 'per_bucket')}, "
                    f"log_topk={mixed_cfg.get('log_topk', 10)}"
                )
            else:
                logger.info("[CONFIG] Mixed-resolution DISABLED")

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
            mixed_cfg = self._get_mixed_resolution_cfg()
            if mixed_cfg.get("enabled", False):
                (
                    samples_batched,
                    images,
                    prompts,
                    last_batch_rewards,
                    last_batch_input_images,
                ) = self._sampling_phase_mixed(epoch, train_iter)

                # Log sample images periodically (same behavior, but images are resized to fixed resolution)
                if epoch % 5 == 0 and rank == 0:
                    self._log_sample_images(
                        images, prompts, {"avg": last_batch_rewards["avg"]}, epoch, last_batch_input_images
                    )

                advantages = self._compute_advantages_mixed(samples_batched)
                if dist.get_rank() == 0 and advantages.numel() > 0:
                    logger.info(
                        f"[mixed_res] advantages: min={advantages.min().item():.6f}, max={advantages.max().item():.6f}, "
                        f"mean={advantages.mean().item():.6f}, std={advantages.std().item():.6f}"
                    )

                self._training_phase_mixed(samples_batched, tokenizer, epoch)
            else:
                samples, images, prompts, last_batch_rewards, last_batch_input_images = self._sampling_phase(
                    epoch, train_iter
                )

                # Add ori_avg for advantage computation (matching train_bagel.py line 742-743)
                samples["rewards"]["ori_avg"] = samples["rewards"]["avg"]
                samples["rewards"]["avg"] = samples["rewards"]["avg"].unsqueeze(-1)

                # Log sample images periodically (matching train_bagel.py line 713-741)
                if epoch % 5 == 0 and rank == 0:
                    rewards_for_logging = {"avg": last_batch_rewards["avg"]}
                    self._log_sample_images(images, prompts, rewards_for_logging, epoch, last_batch_input_images)

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

        tokenizer = self._get_tokenizer()

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
        last_batch_input_images = None  # For image edit tasks

        for batch_idx, batch in enumerate(
            tqdm(
                self.eval_dataloader,
                desc=f"Epoch {epoch}: evaluation",
                disable=dist.get_rank() != 0,
                position=0,
            )
        ):
            prompts, prompt_metadata, input_images = self._parse_rl_prompt_batch(batch)
            if prompts is None:
                logger.warning("Empty prompts in batch; skipping")
                continue

            # Generate images for this batch
            images = []
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                for idx, prompt in enumerate(prompts):
                    # Get input image for this sample (if image edit task)
                    input_image = input_images[idx] if input_images is not None else None

                    with torch.no_grad():
                        think_kwargs = self._get_think_kwargs()
                        inference_hyper = self._get_inference_hyperparams()

                        input_list = []
                        if input_image is not None:
                            input_list.append(input_image)
                        input_list.append(prompt)

                        if think_kwargs.get("think", False):
                            output_list = self.inferencer.interleave_inference(
                                input_list,
                                think=True,
                                understanding_output=False,
                                max_think_token_n=int(think_kwargs.get("max_think_token_n", 1000)),
                                do_sample=bool(think_kwargs.get("do_sample", True)),
                                text_temperature=float(think_kwargs.get("text_temperature", 0.3)),
                                cfg_text_scale=eval_guidance_scale,
                                num_timesteps=eval_num_steps,
                                noise_level=0,
                                grpo_config=self.grpo_config,
                                accelerator=None,
                                **inference_hyper,
                            )
                            output_dict = output_list[1]
                        else:
                            output_list = self.inferencer.interleave_inference(
                                input_list,
                                think=False,
                                understanding_output=False,
                                cfg_text_scale=eval_guidance_scale,
                                num_timesteps=eval_num_steps,
                                noise_level=0,
                                grpo_config=self.grpo_config,
                                accelerator=None,
                                **inference_hyper,
                            )
                            output_dict = output_list[0]
                    # Image is already tensor (C, H, W) in [0, 1], same as flow_grpo
                    images.append(output_dict["image"])

            # Stack images: (batch_size, 3, H, W)
            images = torch.stack(images, dim=0)

            # Compute rewards asynchronously
            # For image editing: pass input_images as ref_images for image_similarity reward
            if input_images is not None:
                rewards_future = self.executor.submit(
                    self.eval_reward_fn, images, prompts, prompt_metadata, ref_images=input_images, only_strict=False
                )
            else:
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
            last_batch_input_images = input_images  # For image edit tasks

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
                # Check if this is an image edit task
                is_image_edit = last_batch_input_images is not None and len(last_batch_input_images) > 0

                with tempfile.TemporaryDirectory() as tmpdir:
                    num_samples = min(25, len(images_gathered))
                    sample_indices = list(range(num_samples))

                    # Save output images to temp directory
                    for idx, index in enumerate(sample_indices):
                        image = images_gathered[index]
                        # image shape: (C, H, W) -> (H, W, C), values in [0, 1]
                        pil = Image.fromarray((image.transpose(1, 2, 0) * 255).astype(np.uint8))
                        pil = pil.resize((self.grpo_config.resolution, self.grpo_config.resolution))
                        pil.save(os.path.join(tmpdir, f"output_{idx}.jpg"))

                    # Save input images for image edit tasks
                    if is_image_edit:
                        for idx, index in enumerate(sample_indices):
                            if index < len(last_batch_input_images) and last_batch_input_images[index] is not None:
                                input_img = last_batch_input_images[index]
                                # Handle different input types
                                if isinstance(input_img, Image.Image):
                                    pil = input_img.convert("RGB")
                                elif isinstance(input_img, torch.Tensor):
                                    if input_img.dim() == 3:  # (C, H, W)
                                        pil = Image.fromarray(
                                            (input_img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                                        )
                                    else:
                                        continue
                                else:
                                    continue
                                pil = pil.resize((self.grpo_config.resolution, self.grpo_config.resolution))
                                pil.save(os.path.join(tmpdir, f"input_{idx}.jpg"))

                    # Prepare prompts and rewards for caption
                    sampled_prompts = [prompts_gathered[i] for i in sample_indices if i < len(prompts_gathered)]
                    sampled_rewards = [
                        {k: last_batch_rewards_gathered[k][i] for k in last_batch_rewards_gathered}
                        for i in sample_indices
                        if i < len(prompts_gathered)
                    ]

                    if is_image_edit:
                        # For image edit: create side-by-side comparison images (input | output)
                        eval_comparison_images = []

                        for idx, (prompt, reward_dict) in enumerate(zip(sampled_prompts, sampled_rewards)):
                            input_path = os.path.join(tmpdir, f"input_{idx}.jpg")
                            output_path = os.path.join(tmpdir, f"output_{idx}.jpg")

                            if os.path.exists(input_path) and os.path.exists(output_path):
                                # Load both images
                                input_img = Image.open(input_path)
                                output_img = Image.open(output_path)

                                # Create side-by-side comparison image
                                width, height = input_img.size
                                combined_width = width * 2 + 10  # 10px gap
                                combined = Image.new("RGB", (combined_width, height), color="white")
                                combined.paste(input_img, (0, 0))
                                combined.paste(output_img, (width + 10, 0))

                                # Save combined image
                                combined_path = os.path.join(tmpdir, f"eval_comparison_{idx}.jpg")
                                combined.save(combined_path)

                                # Build caption with rewards
                                reward_str = " | ".join(
                                    f"{k}: {v:.2f}"
                                    for k, v in reward_dict.items()
                                    if isinstance(v, (int, float)) and v != -10
                                )
                                caption = f"{reward_str} | {prompt[:80]}..."
                                eval_comparison_images.append(wandb.Image(combined_path, caption=caption))

                        # Log all comparison images together
                        if eval_comparison_images:
                            wandb.log(
                                {
                                    "eval_comparisons": eval_comparison_images,
                                    **{
                                        f"eval/reward_{key}": float(np.mean(value[value != -10]))
                                        for key, value in all_rewards_concat.items()
                                        if len(value) > 0
                                    },
                                },
                                step=self.global_step,
                            )
                            logger.info(
                                f"[DEBUG] Logged {len(eval_comparison_images)} eval comparison images (input|output)"
                            )
                    else:
                        # For text-to-image: use original logging format
                        eval_images = [
                            wandb.Image(
                                os.path.join(tmpdir, f"output_{idx}.jpg"),
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

    def _log_sample_images(
        self,
        images: torch.Tensor,
        prompts: List[str],
        rewards: Dict,
        epoch: int,
        input_images: Optional[List] = None,
    ):
        """
        Log sample images to wandb.

        For image edit tasks, displays input_image -> output_image comparison.

        Args:
            images: Generated output images (tensor)
            prompts: Text prompts
            rewards: Reward dict
            epoch: Current epoch
            input_images: Optional list of input PIL Images (for image edit tasks)
        """
        import wandb

        if not hasattr(self, "tracking"):
            return

        num_samples = min(15, len(images))
        if num_samples == 0:
            return

        sample_indices = random.sample(range(len(images)), num_samples)

        # Check if this is an image edit task
        is_image_edit = input_images is not None and len(input_images) > 0

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
            # Save output images
            for idx, i in enumerate(sample_indices):
                image = images[i].cpu().numpy()
                pil = Image.fromarray((image.transpose(1, 2, 0) * 255).astype(np.uint8))
                pil = pil.resize((self.grpo_config.resolution, self.grpo_config.resolution))
                pil.save(os.path.join(tmpdir, f"output_{idx}.jpg"))

            # Save input images for edit tasks
            if is_image_edit:
                for idx, i in enumerate(sample_indices):
                    if i < len(input_images) and input_images[i] is not None:
                        input_img = input_images[i]
                        # Handle different input types
                        if isinstance(input_img, Image.Image):
                            pil = input_img.convert("RGB")
                        elif isinstance(input_img, torch.Tensor):
                            # Convert tensor to PIL
                            if input_img.dim() == 3:  # (C, H, W)
                                pil = Image.fromarray((input_img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
                            else:
                                continue
                        else:
                            continue
                        pil = pil.resize((self.grpo_config.resolution, self.grpo_config.resolution))
                        pil.save(os.path.join(tmpdir, f"input_{idx}.jpg"))

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

            if is_image_edit:
                # For image edit: create side-by-side comparison images (input | output)
                # This makes it easier to compare in wandb without switching panels
                comparison_images = []

                for idx, (prompt, avg_reward) in enumerate(zip(sampled_prompts, sampled_rewards)):
                    input_path = os.path.join(tmpdir, f"input_{idx}.jpg")
                    output_path = os.path.join(tmpdir, f"output_{idx}.jpg")

                    if os.path.exists(input_path) and os.path.exists(output_path):
                        # Load both images
                        input_img = Image.open(input_path)
                        output_img = Image.open(output_path)

                        # Create side-by-side comparison image
                        # Add labels on top
                        width, height = input_img.size
                        label_height = 30
                        combined_width = width * 2 + 10  # 10px gap between images
                        combined_height = height + label_height

                        combined = Image.new("RGB", (combined_width, combined_height), color="white")

                        # Paste input image (left side)
                        combined.paste(input_img, (0, label_height))

                        # Paste output image (right side with gap)
                        combined.paste(output_img, (width + 10, label_height))

                        # Save combined image
                        combined_path = os.path.join(tmpdir, f"comparison_{idx}.jpg")
                        combined.save(combined_path)

                        # Create caption with prompt and reward
                        caption = f"avg: {avg_reward:.2f} | {prompt[:100]}..."
                        comparison_images.append(wandb.Image(combined_path, caption=caption))

                # Log all comparison images together
                if comparison_images:
                    wandb.log(
                        {"edit_comparisons": comparison_images},
                        step=self.global_step,
                    )
                    logger.info(
                        f"[DEBUG] Logged {len(comparison_images)} comparison images (input|output) for image edit task"
                    )
            else:
                # For text-to-image: use original logging format
                self.tracking.log(
                    {
                        "images": [
                            {
                                "image": os.path.join(tmpdir, f"output_{idx}.jpg"),
                                "caption": f"{prompt[:100]} | avg: {avg_reward:.2f}",
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
