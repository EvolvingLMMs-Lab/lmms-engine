import os
import re
from typing import Any, Dict, List, Optional, Tuple

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
        self.args = args
        self._enabled_cache: Optional[bool] = None
        self._initialized: bool = False

        # param_name -> EMA tensor/DTensor
        self._ema_params: Dict[str, Any] = {}
        # (ema_tensor, live_param)
        self._ema_param_pairs: List[Tuple[Any, nn.Parameter]] = []

    @property
    def initialized(self) -> bool:
        return self._initialized

    def is_enabled(self) -> bool:
        if self._enabled_cache is not None:
            return self._enabled_cache

        enabled = bool(getattr(self.args, "ema_enabled", False))
        decay = float(getattr(self.args, "ema_decay", 0.0) or 0.0)
        update_every = int(getattr(self.args, "ema_update_every", 1) or 1)

        if not enabled:
            self._enabled_cache = False
            return False
        if not (0.0 < decay < 1.0):
            logger.warning(f"EMA enabled but ema_decay={decay} is invalid; EMA will be disabled.")
            self._enabled_cache = False
            return False
        if update_every <= 0:
            logger.warning(f"EMA enabled but ema_update_every={update_every} is invalid; EMA will be disabled.")
            self._enabled_cache = False
            return False

        self._enabled_cache = True
        return True

    def maybe_init(self, model: nn.Module, checkpoint_dir: Optional[str]) -> None:
        if self._initialized or (not self.is_enabled()):
            return

        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

        # Prefer EMA checkpoint if present.
        ema_state_dict: Optional[Dict[str, Any]] = None
        if checkpoint_dir is not None:
            ema_model_path = os.path.join(
                checkpoint_dir,
                "pytorch_ema_model_fsdp_0",
                f"model_world_size_{world_size}_rank_{rank}.pt",
            )
            if os.path.exists(ema_model_path):
                try:
                    ema_state_dict = torch.load(ema_model_path, weights_only=False)
                    if rank == 0:
                        logger.info(f"Loaded EMA state from checkpoint: {ema_model_path}")
                except Exception as e:
                    logger.warning(f"Failed to load EMA checkpoint ({ema_model_path}): {e}. Will init EMA from model.")

        for name, param in model.named_parameters():
            if not self._should_track_param(name, param):
                continue

            if ema_state_dict is not None and name in ema_state_dict:
                ema_val = ema_state_dict[name]
            else:
                # Clone shard-local value (Tensor or DTensor)
                ema_val = param.detach().clone()

            self._ema_params[name] = ema_val
            self._ema_param_pairs.append((ema_val, param))

        self._initialized = True

        # Optional: use EMA weights for the live model when resuming.
        if bool(getattr(self.args, "ema_resume_from_ema", False)) and (ema_state_dict is not None):
            self.copy_to_model()
            if rank == 0:
                logger.info("Applied EMA weights to live model (ema_resume_from_ema=True).")

    def update(self, step: int) -> None:
        if (not self.is_enabled()) or (not self._initialized) or (not self._ema_param_pairs):
            return

        start_step = int(getattr(self.args, "ema_start_step", 0) or 0)
        update_every = int(getattr(self.args, "ema_update_every", 1) or 1)
        if step < start_step:
            return
        if update_every > 1 and (step % update_every != 0):
            return

        decay = float(getattr(self.args, "ema_decay", 0.0))
        one_minus_decay = 1.0 - decay

        # Shard-local update to avoid any distributed ops.
        for ema_val, param in self._ema_param_pairs:
            ema_local = self._local_tensor(ema_val)
            model_local = self._local_tensor(param)
            ema_local.mul_(decay).add_(model_local, alpha=one_minus_decay)

    def copy_to_model(self) -> None:
        if (not self._initialized) or (not self._ema_param_pairs):
            return
        for ema_val, param in self._ema_param_pairs:
            ema_local = self._local_tensor(ema_val)
            model_local = self._local_tensor(param)
            model_local.copy_(ema_local)

    def state_dict_for_save(self, model: nn.Module) -> Dict[str, Any]:
        """
        Build a sharded state_dict compatible with the existing FSDP2 checkpoint format,
        but with EMA-smoothed parameter values.
        """
        if not self._initialized:
            raise RuntimeError("EMA is not initialized; cannot build EMA state_dict for save.")

        sd = model.state_dict()
        for name, ema_val in self._ema_params.items():
            # Only override keys that actually exist in the model state dict.
            if name in sd:
                sd[name] = ema_val
        return sd

    @staticmethod
    def _local_tensor(t: Any) -> torch.Tensor:
        # Unwrap nn.Parameter
        if isinstance(t, nn.Parameter):
            t = t.data
        # DTensor / sharded tensor objects used by FSDP2 commonly expose `_local_tensor`
        if hasattr(t, "_local_tensor"):
            return t._local_tensor  # noqa: SLF001 (intentional: avoid expensive distributed ops)
        if isinstance(t, torch.Tensor):
            return t
        raise TypeError(f"Unsupported EMA tensor type: {type(t)}")

    def _name_match(self, name: str, patterns: List[str], mode: str) -> bool:
        if mode == "regex":
            return any(re.search(p, name) is not None for p in patterns)
        # default: substring
        return any(p in name for p in patterns)

    def _should_track_param(self, name: str, param: nn.Parameter) -> bool:
        if bool(getattr(self.args, "ema_requires_grad_only", True)) and (not param.requires_grad):
            return False

        local = self._local_tensor(param)
        if not torch.is_floating_point(local):
            return False

        cfg = getattr(self.args, "ema_param_filter", None) or {}
        mode = str(cfg.get("mode", "substring") or "substring").lower()
        include = cfg.get("include", None)
        exclude = cfg.get("exclude", None)

        if include:
            include = list(include)
            if not self._name_match(name, include, mode):
                return False
        if exclude:
            exclude = list(exclude)
            if self._name_match(name, exclude, mode):
                return False
        return True
