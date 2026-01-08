import os
import re
from typing import Any, Dict, List, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from loguru import logger


class EMAHelper:
    """
    Lightweight Exponential Moving Average (EMA) helper designed to work with:
    - Vanilla nn.Module parameters (torch.Tensor)
    - FSDP2 / sharded parameters that expose a shard-local `_local_tensor`

    Key design goals:
    - Fully opt-in (no behavior change unless ema_enabled=True)
    - Shard-local updates (no all-gather / no distributed communication)
    - Generic for HF-style models (operates on model.named_parameters())
    """

    def __init__(self, args) -> None:
        # Extract all config once - args won't change at runtime
        self._enabled = bool(getattr(args, "ema_enabled", False))
        self._decay = float(getattr(args, "ema_decay", 0.9999))
        self._update_every = int(getattr(args, "ema_update_every", 1))
        self._start_step = int(getattr(args, "ema_start_step", 0))
        self._requires_grad_only = bool(getattr(args, "ema_requires_grad_only", True))
        self._resume_from_ema = bool(getattr(args, "ema_resume_from_ema", False))

        param_filter = getattr(args, "ema_param_filter", None) or {}
        self._param_filter_mode = str(param_filter.get("mode", "substring") or "substring").lower()
        self._param_include = param_filter.get("include")
        self._param_exclude = param_filter.get("exclude")

        # Disable if config is invalid
        if not (0.0 < self._decay < 1.0) or self._update_every <= 0:
            self._enabled = False

        self._initialized: bool = False
        self._ema_params: Dict[str, Any] = {}
        self._ema_param_pairs: List[Tuple[Any, nn.Parameter]] = []

    def is_enabled(self) -> bool:
        return self._enabled

    @property
    def initialized(self) -> bool:
        return self._initialized

    def maybe_init(self, model: nn.Module, checkpoint_dir: str | None) -> None:
        if self._initialized or not self._enabled:
            return

        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

        # Load EMA checkpoint if present
        ema_state_dict = None
        if checkpoint_dir is not None:
            ema_model_path = os.path.join(
                checkpoint_dir,
                "pytorch_ema_model_fsdp_0",
                f"model_world_size_{world_size}_rank_{rank}.pt",
            )
            if os.path.exists(ema_model_path):
                try:
                    ema_state_dict = torch.load(ema_model_path, weights_only=False)
                    logger.info(f"Loaded EMA state from checkpoint: {ema_model_path}")
                except Exception as e:
                    logger.warning(f"Failed to load EMA checkpoint ({ema_model_path}): {e}. Will init EMA from model.")

        for name, param in model.named_parameters():
            if not self._should_track_param(name, param):
                continue

            ema_val = ema_state_dict[name] if (ema_state_dict and name in ema_state_dict) else param.detach().clone()
            self._ema_params[name] = ema_val
            self._ema_param_pairs.append((ema_val, param))

        self._initialized = True

        if self._resume_from_ema and ema_state_dict is not None:
            self.copy_to_model()
            if rank == 0:
                logger.info("Applied EMA weights to live model (ema_resume_from_ema=True).")

    def update(self, step: int) -> None:
        if not self._enabled or not self._initialized:
            return
        if step < self._start_step:
            return
        if self._update_every > 1 and (step % self._update_every != 0):
            return

        one_minus_decay = 1.0 - self._decay
        for ema_val, param in self._ema_param_pairs:
            ema_local = self._local_tensor(ema_val)
            model_local = self._local_tensor(param)
            ema_local.mul_(self._decay).add_(model_local, alpha=one_minus_decay)

    def copy_to_model(self) -> None:
        if not self._initialized:
            return
        for ema_val, param in self._ema_param_pairs:
            ema_local = self._local_tensor(ema_val)
            model_local = self._local_tensor(param)
            model_local.copy_(ema_local)

    def state_dict_for_save(self, model: nn.Module) -> Dict[str, Any]:
        if not self._initialized:
            raise RuntimeError("EMA is not initialized; cannot build EMA state_dict for save.")

        sd = model.state_dict()
        for name, ema_val in self._ema_params.items():
            if name in sd:
                sd[name] = ema_val
        return sd

    @staticmethod
    def _local_tensor(t: Any) -> torch.Tensor:
        if isinstance(t, nn.Parameter):
            t = t.data
        # FSDP2 shards expose _local_tensor for shard access without all-gather
        if hasattr(t, "_local_tensor"):
            return t._local_tensor  # noqa: SLF001
        return t

    def _name_match(self, name: str, patterns: List[str]) -> bool:
        if self._param_filter_mode == "regex":
            return any(re.search(p, name) is not None for p in patterns)
        return any(p in name for p in patterns)

    def _should_track_param(self, name: str, param: nn.Parameter) -> bool:
        if self._requires_grad_only and not param.requires_grad:
            return False

        if not torch.is_floating_point(self._local_tensor(param)):
            return False

        if self._param_include and not self._name_match(name, self._param_include):
            return False
        if self._param_exclude and self._name_match(name, self._param_exclude):
            return False

        return True
