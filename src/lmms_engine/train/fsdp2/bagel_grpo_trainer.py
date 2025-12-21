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
    return getattr(m, "module", m)


@contextmanager
def _temporary_set_train_mode(module: nn.Module, train: bool = True):
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
    generators = []
    for prompt in prompts:
        hash_digest = hashlib.sha256(prompt.encode()).digest()
        prompt_hash_int = int.from_bytes(hash_digest[:4], "big")
        seed = (base_seed + prompt_hash_int) % (2**31)
        gen = torch.Generator().manual_seed(seed)
        generators.append(gen)
    return generators


def calculate_zero_std_ratio(prompts: List[str], gathered_rewards: Dict[str, np.ndarray]) -> tuple:
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
        self.inferencer = kwargs.get("inferencer") or self._create_inferencer_from_model()
        self.reward_fn = kwargs.get("reward_fn") or self._create_reward_fn(device)
        self.eval_reward_fn = kwargs.get("eval_reward_fn") or self.reward_fn

    def _normalize_reward_config(self, cfg):
        from types import SimpleNamespace

        if isinstance(cfg, SimpleNamespace):
            return dict(cfg.__dict__)
        if isinstance(cfg, str):
            return {cfg: 1.0}
        return cfg if isinstance(cfg, dict) else {}

    def _create_reward_fn(self, device):
        import lmms_engine.utils.rewards.rewards

        cfg = self._normalize_reward_config(getattr(self.grpo_config, "reward_fn", {}))
        logger.info(f"Initialized reward function with config: {cfg}")
        return lmms_engine.utils.rewards.rewards.multi_score(device, cfg)

    def _create_inferencer_from_model(self):
        from lmms_engine.models.bagel.inferencer import InterleaveInferencer

        vae_model = getattr(self.model, "vae_model", None)
        if vae_model is None:
            logger.warning("Could not find vae_model in model. Inferencer may not work correctly.")

        return InterleaveInferencer(
            model=self.model,
            vae_model=vae_model,
            tokenizer=self.processing_class.processor,
            vae_transform=self.processing_class.vae_image_transform,
            vit_transform=self.processing_class.vit_image_transform,
            new_token_ids=self.processing_class.new_token_ids,
        )

    def _dict_to_config(self, config_dict):
        from types import SimpleNamespace

        def dict_to_namespace(d):
            if isinstance(d, dict):
                return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
            elif isinstance(d, list):
                return [dict_to_namespace(item) for item in d]
            else:
                return d

        return dict_to_namespace(config_dict)

    def _get_tokenizer(self):
        return getattr(self.processing_class, "processor", self.processing_class)

    def _parse_rl_prompt_batch(self, batch):
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
        super().prepare_model()
        self.fsdp2_model = self.model

        if hasattr(self, "inferencer") and self.inferencer is not None:
            if hasattr(self.inferencer, "model"):
                self.inferencer.model = self.model
            if hasattr(self.model, "vae_model") and hasattr(self.inferencer, "vae_model"):
                self.inferencer.vae_model = self.model.vae_model

        register_fsdp_forward_method(self.model, "forward_cache_update_vae")
        register_fsdp_forward_method(self.model, "forward_cache_update_vit")
        register_fsdp_forward_method(self.model, "forward_cache_update_text")
        register_fsdp_forward_method(self.model, "generate_text")
        register_fsdp_forward_method(self.model, "generate_image")
        register_fsdp_forward_method(self.model, "_forward_flow")
        register_fsdp_forward_method(self.model.vae_model, "decode")
        register_fsdp_forward_method(self.model.vae_model, "encode")

    def prepare_optimizer(self):
        if hasattr(self.model, "language_model"):
            trainable_params = [p for p in self.model.language_model.parameters() if p.requires_grad]
            logger.info(f"Number of trainable parameters: {sum(p.numel() for p in trainable_params)}")
        else:
            trainable_params = [p for p in self.fsdp2_model.parameters() if p.requires_grad]

        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
        )

    def compute_loss(self, batch):
        raise NotImplementedError("GRPO trainer uses custom training_step")

    def training_step(self, batch):
        raise NotImplementedError("Use custom train() method for GRPO")

    def _sampling_phase(self, epoch: int, train_iter):
        self.fsdp2_model.eval()
        samples = []

        num_batches_per_epoch = self.grpo_config.sample.num_batches_per_epoch
        train_batch_size = self.grpo_config.sample.train_batch_size

        if dist.get_rank() == 0:
            logger.info(f"_sampling_phase: epoch={epoch}, num_batches={num_batches_per_epoch}, bs={train_batch_size}")

        if num_batches_per_epoch <= 0:
            logger.error(f"num_batches_per_epoch={num_batches_per_epoch} is invalid!")

        for i in tqdm(
            range(num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
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

            is_image_edit = input_images is not None
            if i == 0 and dist.get_rank() == 0:
                logger.info(f"Task type: {'Image Edit' if is_image_edit else 'Text-to-Image Generation'}")

            tokenizer = self._get_tokenizer()
            prompt_ids = self._tokenize_prompts(prompts, tokenizer)
            self.total_tokens += self._calculate_tokens_for_batch(prompt_ids, num_images=1)
            generators = create_generators(prompts, base_seed=42)

            images = []
            latents = []
            log_probs = []
            timesteps = []
            think_texts: List[Optional[str]] = []

            autocast_dtype = torch.bfloat16 if self.args.bf16 else torch.float16
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                for idx, prompt in enumerate(prompts):
                    if self.grpo_config.sample.same_latent:
                        generator = generators[idx : idx + 1]
                    else:
                        generator = None

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

                    images.append(output_dict["image"])
                    latents.append(output_dict["all_latents"])
                    log_probs.append(output_dict["all_log_probs"])
                    timesteps.append(output_dict["timesteps"])
                    think_texts.append(think_text)

            stacked_inner_latents = [torch.stack(inner_list, dim=0) for inner_list in latents]
            latents = torch.stack(stacked_inner_latents, dim=0)
            stacked_inner_log_probs = [torch.stack(inner_list, dim=0) for inner_list in log_probs]
            log_probs = torch.stack(stacked_inner_log_probs, dim=0)
            timesteps = torch.stack(timesteps, dim=0)
            images = torch.stack(images, dim=0)

            if input_images is not None:
                rewards = self.executor.submit(
                    self.reward_fn, images, prompts, prompt_metadata, ref_images=input_images, only_strict=True
                )
            else:
                rewards = self.executor.submit(self.reward_fn, images, prompts, prompt_metadata, only_strict=True)
            time.sleep(0)

            sample_dict = {
                "prompt_ids": prompt_ids,
                "timesteps": timesteps,
                "latents": latents[:, :-1],
                "prev_latents": latents[:, 1:],
                "log_probs": log_probs,
                "rewards": rewards,
            }

            if input_images is not None:
                sample_dict["input_images"] = input_images

            if any(t is not None for t in think_texts):
                sample_dict["think_texts"] = think_texts

            samples.append(sample_dict)

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

            batch_size = sample["prompt_ids"].shape[0]
            rewards["avg"] = self._normalize_rewards_avg_list(rewards, batch_size)
            sample["rewards"] = self._rewards_to_tensor_dict(rewards)

        samples_collated = {}
        for k in samples[0].keys():
            if k == "input_images":
                all_images = []
                for s in samples:
                    if "input_images" in s and s["input_images"] is not None:
                        all_images.extend(s["input_images"])
                samples_collated[k] = all_images if all_images else None
            elif k == "think_texts":
                all_think = []
                for s in samples:
                    vals = s.get("think_texts", None)
                    if vals is not None:
                        all_think.extend(list(vals))
                samples_collated[k] = all_think if all_think else None
            elif isinstance(samples[0][k], dict):
                samples_collated[k] = {
                    sub_key: torch.cat([s[k][sub_key] for s in samples], dim=0) for sub_key in samples[0][k]
                }
            elif isinstance(samples[0][k], torch.Tensor):
                samples_collated[k] = torch.cat([s[k] for s in samples], dim=0)
            else:
                samples_collated[k] = [s[k] for s in samples]

        last_batch_rewards = samples[-1]["rewards"]
        last_batch_input_images = samples[-1].get("input_images", None)

        return samples_collated, images, prompts, last_batch_rewards, last_batch_input_images

    def _compute_advantages(self, samples: Dict, prompts: List[str], tokenizer):
        gathered_rewards = {}
        for key, value in samples["rewards"].items():
            gathered_value = self._gather_tensor(value)
            gathered_rewards[key] = gathered_value.cpu().numpy()

        if "avg" in gathered_rewards:
            logger.info(f"Gathered rewards['avg'] shape: {gathered_rewards['avg'].shape}")

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

        # Optional: multi-reward aggregation functions (PaCo-style), YAML-configurable.
        # When enabled, the aggregate_fn returns advantages directly, so we skip stat_tracker normalization.
        from lmms_engine.utils.rewards.aggregation import (
            build_aggregate_fn,
            group_rewards_by_prompt,
            map_prompt_advantages_to_order,
        )

        reward_weights = self._normalize_reward_config(getattr(self.grpo_config, "reward_fn", {}))
        train_cfg = getattr(self.grpo_config, "train", None)
        agg_spec = (
            getattr(train_cfg, "aggregate_fn", None)
            if train_cfg is not None
            else getattr(self.grpo_config, "aggregate_fn", None)
        )
        agg_kwargs = (
            getattr(train_cfg, "aggregate_fn_kwargs", None)
            if train_cfg is not None
            else getattr(self.grpo_config, "aggregate_fn_kwargs", None)
        )
        default_global_std = bool(getattr(getattr(self.grpo_config, "sample", None), "global_std", False))
        aggregate_fn = build_aggregate_fn(
            agg_spec, agg_kwargs, reward_weights=reward_weights, default_global_std=default_global_std
        )

        need_prompts = (self.stat_tracker is not None) or (aggregate_fn is not None)
        prompts_decoded: Optional[List[str]] = None
        if need_prompts:
            prompt_ids = self._gather_tensor(samples["prompt_ids"]).cpu().numpy()
            prompts_decoded = tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)

        if aggregate_fn is not None:
            assert prompts_decoded is not None
            # Only aggregate rewards that are explicitly configured in reward_fn.
            rewards_for_agg: Dict[str, np.ndarray] = {}
            for rname in reward_weights.keys():
                if rname not in gathered_rewards:
                    logger.warning(
                        f"[aggregate_fn] reward '{rname}' not found in gathered_rewards. Available: {list(gathered_rewards.keys())}"
                    )
                    continue
                arr = gathered_rewards[rname]
                # Ensure (N, D) for downstream reshape logic
                if isinstance(arr, np.ndarray) and arr.ndim == 1:
                    arr = arr[:, None]
                rewards_for_agg[rname] = arr

            if len(rewards_for_agg) == 0:
                logger.warning(
                    "[aggregate_fn] No valid rewards to aggregate; falling back to default advantages from rewards['avg']."
                )
                aggregate_fn = None
            else:
                grouped = group_rewards_by_prompt(prompts_decoded, rewards_for_agg)
                per_prompt_adv = aggregate_fn(**grouped)
                advantages = map_prompt_advantages_to_order(prompts_decoded, per_prompt_adv)
        if aggregate_fn is None and self.stat_tracker is not None:
            assert prompts_decoded is not None
            advantages = self.stat_tracker.update(prompts_decoded, gathered_rewards["avg"])

            if isinstance(advantages, np.ndarray):
                if np.all(advantages == 0) or np.std(advantages) < 1e-6:
                    logger.warning(
                        "All advantages are zero or have zero std in per-prompt normalization. Using raw rewards minus mean."
                    )
                    advantages = gathered_rewards["avg"] - gathered_rewards["avg"].mean()

            if dist.get_rank() == 0:
                group_size, trained_prompt_num = self.stat_tracker.get_stats()
                zero_std_ratio, reward_std_mean = calculate_zero_std_ratio(prompts_decoded, gathered_rewards)
                if hasattr(self, "tracking"):
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
            reward_std = gathered_rewards["avg"].std()
            if reward_std == 0 or reward_std < 1e-6:
                logger.warning(f"Reward std is zero or very small ({reward_std}), using raw rewards as advantages")
                advantages = gathered_rewards["avg"] - gathered_rewards["avg"].mean()
            else:
                advantages = (gathered_rewards["avg"] - gathered_rewards["avg"].mean()) / (reward_std + 1e-4)

        advantages = torch.as_tensor(advantages)
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        advantages = advantages.reshape(world_size, -1, advantages.shape[-1])[rank]
        advantages = advantages.to(self.fsdp2_model.device)

        if torch.isnan(advantages).any() or torch.isinf(advantages).any():
            logger.error(f"NaN/Inf detected in advantages! shape: {advantages.shape}")
            advantages = torch.where(
                torch.isnan(advantages) | torch.isinf(advantages), torch.zeros_like(advantages), advantages
            )
        else:
            logger.info(
                f"Advantages stats: min={advantages.min()}, max={advantages.max()}, mean={advantages.mean()}, std={advantages.std()}"
            )

        return advantages

    def _force_inference_mode(self, module):
        if module.training:
            module.training = False
        for child in module.children():
            self._force_inference_mode(child)

    def _training_phase(self, samples: Dict, advantages: torch.Tensor, tokenizer, epoch: int):
        self.fsdp2_model.train()

        # Force inference mode on language_model (recursively sets training=False)
        # This is needed because Qwen2 modules dispatch differently based on self.training
        lm = getattr(self.model, "language_model", None)
        if lm is not None:
            self._force_inference_mode(lm)

        total_batch_size, num_timesteps = samples["timesteps"].shape
        num_inner_epochs = self.grpo_config.train.num_inner_epochs
        num_batches_per_epoch = self.grpo_config.sample.num_batches_per_epoch

        if dist.get_rank() == 0:
            rebatch_size = total_batch_size // num_batches_per_epoch if num_batches_per_epoch > 0 else 0
            logger.info(
                f"_training_phase: epoch={epoch}, total_bs={total_batch_size}, "
                f"rebatch_size={rebatch_size}, global_step={self.global_step}"
            )
            if rebatch_size == 0:
                logger.error(f"rebatch_size is 0! total_bs={total_batch_size}, num_batches={num_batches_per_epoch}")

        has_input_images = "input_images" in samples and samples["input_images"] is not None
        input_images_list = samples.get("input_images", None)
        has_think_texts = "think_texts" in samples and samples["think_texts"] is not None
        think_texts_list = samples.get("think_texts", None)

        for inner_epoch in range(num_inner_epochs):
            rebatch_size = total_batch_size // num_batches_per_epoch
            samples_batched = {}
            for k, v in samples.items():
                if k in ("input_images", "think_texts"):
                    continue
                elif isinstance(v, torch.Tensor):
                    samples_batched[k] = v.reshape(-1, rebatch_size, *v.shape[1:])
                else:
                    continue

            samples_batched = [dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())]

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
                sample["dtimesteps"] = torch.cat(
                    [sample["timesteps"][:, :-1] - sample["timesteps"][:, 1:], sample["timesteps"][:, -1].unsqueeze(1)],
                    dim=1,
                )

                bs = sample["timesteps"].shape[0]
                prompts = tokenizer.batch_decode(sample["prompt_ids"], skip_special_tokens=True)
                self.total_tokens += self._calculate_tokens_for_batch(sample["prompt_ids"], num_images=1)

                batch_input_images = sample.get("input_images", None)
                batch_think_texts = sample.get("think_texts", None)

                if batch_think_texts is not None and "advantages" in sample:
                    try:
                        adv_vec = sample["advantages"]
                        if isinstance(adv_vec, torch.Tensor):
                            adv_vec = adv_vec.view(-1)
                        self._reinforce_think(
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
                    cur_sample = {}
                    for k, v in sample.items():
                        if k in ("input_images", "think_texts"):
                            continue
                        elif k == "advantages":
                            if v.dim() == 1:
                                cur_sample[k] = v[j : j + 1]
                            else:
                                cur_sample[k] = v[j]
                        elif isinstance(v, torch.Tensor):
                            cur_sample[k] = v[j]

                    input_image = batch_input_images[j] if batch_input_images is not None else None

                    num_steps = self.grpo_config.sample.num_steps
                    timestep_shift = self.grpo_config.train.timestep_shift
                    device = cur_sample["timesteps"].device

                    grid = torch.linspace(1, 0, num_steps, device=device, dtype=torch.float32)
                    grid = timestep_shift * grid / (1 + (timestep_shift - 1) * grid)

                    sampled_timesteps = cur_sample["timesteps"]
                    indices = torch.argmin(torch.abs(sampled_timesteps.unsqueeze(-1) - grid.unsqueeze(0)), dim=-1)
                    cur_sample["timesteps"] = grid[indices]

                    autocast_dtype = torch.bfloat16 if self.args.bf16 else torch.float16
                    with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                        inference_hyper = self._get_inference_hyperparams()
                        think_kwargs = self._get_think_kwargs()

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

                if self._should_sync_gradients():
                    info_aggregated = {
                        k: torch.mean(torch.stack(v).to(self.fsdp2_model.device)) for k, v in info.items()
                    }
                    for key in info_aggregated:
                        info_aggregated[key] = self._reduce_tensor(info_aggregated[key])

                    info_aggregated.update({"epoch": epoch, "inner_epoch": inner_epoch})

                    total_tokens_tensor = torch.tensor(
                        self.total_tokens, device=self.fsdp2_model.device, dtype=torch.long
                    )
                    dist.all_reduce(total_tokens_tensor, op=dist.ReduceOp.SUM)
                    self.total_tokens = total_tokens_tensor.item()

                    if dist.get_rank() == 0:
                        from lmms_engine.utils import TrainUtilities

                        info_aggregated["train/total_tokens"] = TrainUtilities.format_tokens(self.total_tokens)

                        info_aggregated_for_logging = {}
                        for key, value in info_aggregated.items():
                            if isinstance(value, torch.Tensor):
                                if value.numel() == 1:
                                    info_aggregated_for_logging[key] = value.item()
                                else:
                                    logger.warning(f"Skipping multi-element tensor for key {key}: shape {value.shape}")
                            elif isinstance(value, (int, float, np.integer, np.floating)):
                                info_aggregated_for_logging[key] = (
                                    float(value) if isinstance(value, (np.integer, np.floating)) else value
                                )
                            elif isinstance(value, str):
                                continue
                            else:
                                try:
                                    info_aggregated_for_logging[key] = float(value)
                                except (ValueError, TypeError):
                                    logger.warning(f"Skipping non-scalar value for key {key}: type {type(value)}")

                        logger.info(
                            f"Logging at step={self.global_step}: "
                            f"policy_loss={info_aggregated_for_logging.get('policy_loss', 'N/A'):.6f}, "
                            f"loss={info_aggregated_for_logging.get('loss', 'N/A')}"
                        )

                        if hasattr(self, "tracking"):
                            self.tracking.log(info_aggregated_for_logging, step=self.global_step)

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

    def _reinforce_think(
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
        current_value = getattr(self.grpo_config.sample, "num_batches_per_epoch", None)
        if current_value is not None and current_value != -1:
            return None

        # Get dataset size
        if isinstance(self.train_dataset, IterableDataset):
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

        effective_batch_size = train_batch_size * world_size
        num_batches = (dataset_size + effective_batch_size - 1) // effective_batch_size

        num_batches = max(1, num_batches)
        if dist.get_rank() == 0:
            logger.info(f"Auto-calculated num_batches_per_epoch={num_batches} (dataset_size={dataset_size})")
        return num_batches

    def train(self, resume_from_checkpoint: bool = False):
        self.prepare_model()
        train_dataloader = self.prepare_dataloader(self.train_dataset, is_training=True)
        self.train_dataloader = train_dataloader

        current_value = getattr(self.grpo_config.sample, "num_batches_per_epoch", None)
        if current_value is None or current_value == -1:
            auto_calculated = self._auto_calculate_num_batches_per_epoch()
            self.grpo_config.sample.num_batches_per_epoch = auto_calculated if auto_calculated else 10

        if self.grpo_config.sample.num_batches_per_epoch <= 0:
            self.grpo_config.sample.num_batches_per_epoch = 10
            logger.warning("num_batches_per_epoch was <= 0, forced to 10")

        self.prepare_optimizer()
        self.prepare_and_validate_config()

        warmup_steps = (
            int(self.total_steps * self.args.warmup_ratio) if self.args.warmup_ratio > 0 else self.args.warmup_steps
        )
        self.prepare_scheduler(warmup_steps, self.total_steps)

        rank = dist.get_rank()
        world_size = dist.get_world_size()

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

        for epoch in range(start_epoch, self.args.num_train_epochs):
            if dist.get_rank() == 0:
                logger.info(f"Starting epoch {epoch}/{self.args.num_train_epochs}, global_step={self.global_step}")

            if (
                hasattr(self.grpo_config, "eval_freq")
                and epoch % self.grpo_config.eval_freq == 0
                and epoch > 0
                and self.eval_dataset is not None
            ):
                self._eval_phase(epoch)

            if self.should_save:
                output_dir = os.path.join(self.args.output_dir, f"checkpoint-{self.global_step}")
                self.save_checkpoints(output_dir, self.global_step, total_limit=self.args.save_total_limit)

            samples, images, prompts, last_batch_rewards, last_batch_input_images = self._sampling_phase(
                epoch, train_iter
            )

            samples["rewards"]["ori_avg"] = samples["rewards"]["avg"]
            samples["rewards"]["avg"] = samples["rewards"]["avg"].unsqueeze(-1)

            if epoch % 5 == 0 and rank == 0:
                self._log_sample_images(
                    images, prompts, {"avg": last_batch_rewards["avg"]}, epoch, last_batch_input_images
                )

            advantages = self._compute_advantages(samples, prompts, tokenizer)
            samples["advantages"] = advantages
            del samples["rewards"]

            if dist.get_rank() == 0:
                logger.info(
                    f"Advantages: shape={advantages.shape}, min={advantages.min():.6f}, max={advantages.max():.6f}, "
                    f"mean={advantages.mean():.6f}, std={advantages.std():.6f}"
                )
                if advantages.std().item() < 1e-6:
                    logger.warning("Advantages have near-zero std! Training may not be effective.")

            self._training_phase(samples, advantages, tokenizer, epoch)

            if dist.get_rank() == 0:
                logger.info(f"Completed epoch {epoch}, global_step={self.global_step}")
            if (
                hasattr(self.args, "torch_empty_cache_steps")
                and self.args.torch_empty_cache_steps is not None
                and self.global_step % self.args.torch_empty_cache_steps == 0
            ):
                self.empty_cache()

        # Save final checkpoint
        output_dir = os.path.join(self.args.output_dir, f"checkpoint-{self.global_step}")
        self.save_checkpoints(output_dir, self.global_step, total_limit=self.args.save_total_limit)

    def _to_pil(self, img, resolution: int = None) -> Optional[Image.Image]:
        """Convert tensor/array/PIL to PIL Image, optionally resize."""
        if img is None:
            return None
        if isinstance(img, Image.Image):
            pil = img.convert("RGB")
        elif isinstance(img, torch.Tensor):
            if img.dim() != 3:
                return None
            pil = Image.fromarray((img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
        elif isinstance(img, np.ndarray):
            pil = Image.fromarray((img.transpose(1, 2, 0) * 255).astype(np.uint8))
        else:
            return None
        if resolution:
            pil = pil.resize((resolution, resolution))
        return pil

    def _create_comparison_image(self, input_pil: Image.Image, output_pil: Image.Image, label_height: int = 0):
        """Create side-by-side comparison image (input | output)."""
        w, h = input_pil.size
        combined = Image.new("RGB", (w * 2 + 10, h + label_height), color="white")
        combined.paste(input_pil, (0, label_height))
        combined.paste(output_pil, (w + 10, label_height))
        return combined

    def _format_reward_caption(self, reward_dict_or_val, prompt: str, max_len: int = 100) -> str:
        """Format reward and prompt into caption string."""
        if isinstance(reward_dict_or_val, dict):
            reward_str = " | ".join(
                f"{k}: {v:.2f}" for k, v in reward_dict_or_val.items() if isinstance(v, (int, float)) and v != -10
            )
            return f"{reward_str} | {prompt[:max_len]}..."
        return f"avg: {reward_dict_or_val:.2f} | {prompt[:max_len]}..."

    def _gather_rewards_dict(self, rewards: Dict) -> Dict:
        """Gather reward dict values across processes."""
        result = {}
        for key, value in rewards.items():
            tensor = (
                value.to(self.fsdp2_model.device)
                if isinstance(value, torch.Tensor)
                else torch.as_tensor(value, device=self.fsdp2_model.device)
            )
            result[key] = self._gather_tensor(tensor).cpu().numpy()
        return result

    def _eval_phase(self, epoch: int):
        """Evaluation phase: generate images, compute rewards, log to wandb."""
        import wandb

        if self.eval_dataset is None:
            logger.warning(f"Evaluation skipped at epoch {epoch}: no eval_dataset")
            return

        logger.info(f"Evaluation at epoch {epoch}, global_step={self.global_step}")
        self.fsdp2_model.eval()
        if hasattr(self.model, "language_model"):
            self._force_inference_mode(self.model.language_model)

        if not hasattr(self, "eval_dataloader") or self.eval_dataloader is None:
            self.eval_dataloader = self.prepare_dataloader(self.eval_dataset, is_training=False)

        tokenizer = self._get_tokenizer()
        all_rewards = defaultdict(list)
        autocast_dtype = torch.bfloat16 if self.args.bf16 else torch.float16
        eval_num_steps = getattr(self.grpo_config.sample, "eval_num_steps", 50)
        eval_guidance_scale = getattr(self.grpo_config.sample, "eval_guidance_scale", 4.0)

        last_batch = {"images": None, "prompts": None, "rewards": None, "input_images": None}

        for batch_idx, batch in enumerate(
            tqdm(self.eval_dataloader, desc=f"Epoch {epoch}: eval", disable=dist.get_rank() != 0)
        ):
            prompts, prompt_metadata, input_images = self._parse_rl_prompt_batch(batch)
            if not prompts:
                continue

            images = []
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                for idx, prompt in enumerate(prompts):
                    input_image = input_images[idx] if input_images else None
                    with torch.no_grad():
                        think_kwargs = self._get_think_kwargs()
                        inference_hyper = self._get_inference_hyperparams()
                        input_list = ([input_image] if input_image else []) + [prompt]

                        common_kwargs = dict(
                            understanding_output=False,
                            cfg_text_scale=eval_guidance_scale,
                            num_timesteps=eval_num_steps,
                            noise_level=0,
                            grpo_config=self.grpo_config,
                            accelerator=None,
                            **inference_hyper,
                        )
                        if think_kwargs.get("think", False):
                            output_list = self.inferencer.interleave_inference(
                                input_list,
                                think=True,
                                max_think_token_n=int(think_kwargs.get("max_think_token_n", 1000)),
                                do_sample=bool(think_kwargs.get("do_sample", True)),
                                text_temperature=float(think_kwargs.get("text_temperature", 0.3)),
                                **common_kwargs,
                            )
                            output_dict = output_list[1]
                        else:
                            output_list = self.inferencer.interleave_inference(input_list, think=False, **common_kwargs)
                            output_dict = output_list[0]
                    images.append(output_dict["image"])

            images = torch.stack(images, dim=0)

            reward_kwargs = {"ref_images": input_images} if input_images else {}
            rewards_future = self.executor.submit(
                self.eval_reward_fn, images, prompts, prompt_metadata, only_strict=False, **reward_kwargs
            )
            time.sleep(0)

            try:
                rewards, _ = rewards_future.result()
            except Exception as e:
                logger.error(f"Eval rewards error batch {batch_idx}: {e}")
                rewards = {"avg": [0.0] * len(prompts)}

            for key, value in rewards.items():
                all_rewards[key].append(self._gather_rewards_dict({key: value})[key])

            last_batch = {"images": images, "prompts": prompts, "rewards": rewards, "input_images": input_images}

        all_rewards_concat = {k: np.concatenate(v) for k, v in all_rewards.items()}

        if last_batch["images"] is not None and len(last_batch["images"]) > 0:
            images_gathered = self._gather_tensor(last_batch["images"].to(self.fsdp2_model.device)).cpu().numpy()
            prompt_ids = tokenizer(
                last_batch["prompts"], padding="max_length", max_length=256, truncation=True, return_tensors="pt"
            ).input_ids.to(self.fsdp2_model.device)
            prompts_gathered = tokenizer.batch_decode(
                self._gather_tensor(prompt_ids).cpu().numpy(), skip_special_tokens=True
            )
            rewards_gathered = self._gather_rewards_dict(last_batch["rewards"])
        else:
            images_gathered, prompts_gathered, rewards_gathered = np.array([]), [], {}

        if dist.get_rank() == 0:
            eval_metrics = {
                "eval/epoch": epoch,
                **{
                    f"eval/reward_{k}": float(np.mean(v[v != -10])) for k, v in all_rewards_concat.items() if len(v) > 0
                },
            }
            if hasattr(self, "tracking"):
                self.tracking.log(eval_metrics, step=self.global_step)
            logger.info(f"Eval metrics: {eval_metrics}")

            if len(images_gathered) > 0:
                self._log_eval_images_to_wandb(
                    images_gathered, prompts_gathered, rewards_gathered, last_batch["input_images"], all_rewards_concat
                )

        logger.info(f"Evaluation completed at epoch {epoch}")

    def _log_eval_images_to_wandb(self, images, prompts, rewards_dict, input_images, all_rewards_concat):
        """Log evaluation images to wandb (rank 0 only)."""
        import wandb

        resolution = self.grpo_config.resolution
        is_edit = input_images is not None and len(input_images) > 0
        num_samples = min(25, len(images))

        with tempfile.TemporaryDirectory() as tmpdir:
            sampled_prompts = prompts[:num_samples]
            sampled_rewards = [
                {k: rewards_dict[k][i] for k in rewards_dict} for i in range(num_samples) if i < len(prompts)
            ]

            if is_edit:
                wandb_images = []
                for idx in range(num_samples):
                    output_pil = self._to_pil(images[idx], resolution)
                    input_pil = self._to_pil(input_images[idx], resolution) if idx < len(input_images) else None
                    if output_pil and input_pil:
                        combined = self._create_comparison_image(input_pil, output_pil)
                        path = os.path.join(tmpdir, f"cmp_{idx}.jpg")
                        combined.save(path)
                        caption = self._format_reward_caption(
                            sampled_rewards[idx] if idx < len(sampled_rewards) else {},
                            sampled_prompts[idx] if idx < len(sampled_prompts) else "",
                            80,
                        )
                        wandb_images.append(wandb.Image(path, caption=caption))
                if wandb_images:
                    wandb.log(
                        {
                            "eval_comparisons": wandb_images,
                            **{
                                f"eval/reward_{k}": float(np.mean(v[v != -10]))
                                for k, v in all_rewards_concat.items()
                                if len(v) > 0
                            },
                        },
                        step=self.global_step,
                    )
                    logger.info(f"Logged {len(wandb_images)} eval comparison images")
            else:
                wandb_images = []
                for idx in range(num_samples):
                    pil = self._to_pil(images[idx], resolution)
                    if pil:
                        path = os.path.join(tmpdir, f"out_{idx}.jpg")
                        pil.save(path)
                        caption = self._format_reward_caption(
                            sampled_rewards[idx] if idx < len(sampled_rewards) else {},
                            sampled_prompts[idx] if idx < len(sampled_prompts) else "",
                            100,
                        )
                        wandb_images.append(wandb.Image(path, caption=caption))
                if wandb_images:
                    wandb.log(
                        {
                            "eval_images": wandb_images,
                            **{
                                f"eval/reward_{k}": float(np.mean(v[v != -10]))
                                for k, v in all_rewards_concat.items()
                                if len(v) > 0
                            },
                        },
                        step=self.global_step,
                    )
                    logger.info(f"Logged {len(wandb_images)} eval images")

    def _log_sample_images(
        self, images: torch.Tensor, prompts: List[str], rewards: Dict, epoch: int, input_images: Optional[List] = None
    ):
        """Log sample images to wandb during training."""
        import wandb

        if not hasattr(self, "tracking") or len(images) == 0:
            return

        num_samples = min(15, len(images))
        sample_indices = random.sample(range(len(images)), num_samples)
        is_edit = input_images is not None and len(input_images) > 0
        resolution = self.grpo_config.resolution

        avg_rewards = rewards.get("avg", torch.zeros(len(images)))
        if isinstance(avg_rewards, torch.Tensor) and avg_rewards.dim() == 0:
            avg_rewards = avg_rewards.unsqueeze(0)

        def get_reward(i):
            if i < len(avg_rewards):
                v = avg_rewards[i]
                return v.item() if isinstance(v, torch.Tensor) else v
            return 0.0

        with tempfile.TemporaryDirectory() as tmpdir:
            sampled_prompts = [prompts[i] for i in sample_indices]
            sampled_rewards = [get_reward(i) for i in sample_indices]

            if is_edit:
                wandb_images = []
                for idx, (si, prompt, reward) in enumerate(zip(sample_indices, sampled_prompts, sampled_rewards)):
                    output_pil = self._to_pil(images[si].cpu(), resolution)
                    input_pil = self._to_pil(input_images[si], resolution) if si < len(input_images) else None
                    if output_pil and input_pil:
                        combined = self._create_comparison_image(input_pil, output_pil, label_height=30)
                        path = os.path.join(tmpdir, f"cmp_{idx}.jpg")
                        combined.save(path)
                        wandb_images.append(wandb.Image(path, caption=self._format_reward_caption(reward, prompt)))
                if wandb_images:
                    wandb.log({"edit_comparisons": wandb_images}, step=self.global_step)
                    logger.info(f"Logged {len(wandb_images)} comparison images")
            else:
                self.tracking.log(
                    {
                        "images": [
                            {
                                "image": self._to_pil(images[si].cpu(), resolution),
                                "caption": self._format_reward_caption(reward, prompt),
                            }
                            for si, prompt, reward in zip(sample_indices, sampled_prompts, sampled_rewards)
                            if self._to_pil(images[si].cpu(), resolution)
                        ]
                    },
                    step=self.global_step,
                )
