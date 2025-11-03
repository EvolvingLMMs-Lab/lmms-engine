# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# This code is inspired by the torchtune.
# https://github.com/pytorch/torchtune/blob/main/torchtune/utils/_device.py
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license in https://github.com/pytorch/torchtune/blob/main/LICENSE

import math

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor, Replicate, distribute_tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LinearLR

import lmms_engine.parallel.process_group_manager as pgm


def apply_fsdp2(model, fsdp_kwargs, fsdp_transformer_layer_cls_to_wrap=None):
    """model: AutoModelForCausalLM"""

    default_transformer_cls_names_to_wrap = getattr(model, "_no_split_modules", None)
    fsdp_transformer_layer_cls_to_wrap = (
        default_transformer_cls_names_to_wrap
        if fsdp_transformer_layer_cls_to_wrap is None
        else fsdp_transformer_layer_cls_to_wrap
    )

    if isinstance(fsdp_transformer_layer_cls_to_wrap, str):
        fsdp_transformer_layer_cls_to_wrap = [fsdp_transformer_layer_cls_to_wrap]

    assert len(fsdp_transformer_layer_cls_to_wrap) > 0 and fsdp_transformer_layer_cls_to_wrap[0] is not None

    modules = []
    for name, module in model.named_modules():
        if module.__class__.__name__ in fsdp_transformer_layer_cls_to_wrap or (
            isinstance(module, nn.Embedding) and not model.config.tie_word_embeddings
        ):
            modules.append(module)

    for idx, module in enumerate(modules):
        fully_shard(module, **fsdp_kwargs)
    fully_shard(model, **fsdp_kwargs)  # fsdp2 will not reshard_after_forward for root module


def fsdp2_load_full_state_dict(model: torch.nn.Module, full_state: dict, device_mesh=None, cpu_offload=None):
    """
    Loads the full state dict (could be only on rank 0) into the sharded model. This is done by broadcasting the
    parameters from rank 0 to all other ranks. This function modifies the model in-place.

    Args:
        model (`torch.nn.Module`): The model to load the state dict into
        full_state (`dict`): The full state dict to load, can only be on rank 0
    """
    # If we are applying other parallelism, load state dict in sharded way
    # TODO: add logic for other parallelism
    if pgm.process_group_manager.enable_parallel:
        meta_sharded_sd = model.state_dict()
        sharded_sd = {}
        for param_name, full_tensor in full_state.items():
            sharded_meta_param = meta_sharded_sd.get(param_name)
            if isinstance(full_tensor, DTensor):
                full_tensor = full_tensor.redistribute(
                    device_mesh=full_tensor.device_mesh,
                    placements=[Replicate()] * len(full_tensor.placements),
                ).to_local()
            if isinstance(sharded_meta_param, DTensor):
                sharded_tensor = distribute_tensor(
                    full_tensor,
                    sharded_meta_param.device_mesh,
                    sharded_meta_param.placements,
                )
            else:
                sharded_tensor = full_tensor
            sharded_sd[param_name] = nn.Parameter(sharded_tensor)
        model.load_state_dict(sharded_sd, assign=True)
    else:
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            set_model_state_dict,
        )

        # To broadcast, it needs to be instantiated in the GPU.
        if dist.get_rank() == 0:
            model = model.to(device=torch.cuda.current_device(), non_blocking=True)
        else:
            model = model.to_empty(device=torch.cuda.current_device())

        cpu_offload = cpu_offload is not None
        options = StateDictOptions(full_state_dict=True, cpu_offload=cpu_offload, broadcast_from_rank0=True)
        set_model_state_dict(model, full_state, options=options)

        # rotary_emb is not in state_dict, so we need to broadcast it manually
        for name, buf in model.named_buffers():
            dist.broadcast(buf, src=0)

        if cpu_offload:
            model.to("cpu", non_blocking=True)
            for buf in model.buffers():
                buf.data = buf.data.to(torch.cuda.current_device())


"""
Optimizer related
"""


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        min_lr_ratio (:obj:`float`, `optional`, defaults to 0.0):
            The minimum lr ratio w.r.t the maximum.
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    min_lr_ratio = 0.0 if min_lr_ratio is None else min_lr_ratio
    assert min_lr_ratio >= 0 and min_lr_ratio <= 1.0
    coef = (1 - min_lr_ratio) * 0.5
    intercept = (1 + min_lr_ratio) * 0.5

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return min_lr_ratio + (1.0 - min_lr_ratio) * (float(current_step) / float(max(1, num_warmup_steps)))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        x = math.cos(math.pi * float(num_cycles) * 2.0 * progress)
        return max(min_lr_ratio, x * coef + intercept)

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_wsd_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
    stable_ratio: float = 0.9,
):
    """
    Create a Warmup-Stable-Decay learning rate scheduler.

    The schedule follows three phases:
    1. Warmup: Learning rate increases linearly from 0 to the initial LR
    2. Stable: Learning rate remains constant at the initial LR
    3. Decay: Learning rate decreases following a cosine curve to min_lr_ratio * initial LR

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        min_lr_ratio (:obj:`float`, `optional`, defaults to 0.0):
            The minimum learning rate ratio w.r.t the initial learning rate.
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
            The number of waves in the cosine schedule during decay phase.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.
        stable_ratio (:obj:`float`, `optional`, defaults to 0.0):
            The ratio of non-warmup steps that should maintain a constant learning rate.
            Set to 0.0 to behave exactly like cosine schedule.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    remaining_steps = max(0, num_training_steps - num_warmup_steps)
    num_stable_steps = int(remaining_steps * stable_ratio)
    num_decay_steps = remaining_steps - num_stable_steps

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        if current_step < num_warmup_steps + num_stable_steps:
            return 1.0
        if current_step < num_training_steps:
            progress = float(current_step - num_warmup_steps - num_stable_steps) / float(max(1, num_decay_steps))
            value = max(
                0.0,
                0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)),
            )
            return (1.0 - min_lr_ratio) * value + min_lr_ratio
        return min_lr_ratio

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_constant_schedule(
    optimizer: Optimizer,
    num_warmup_steps: int,
    start_factor: float = 0.01,
    end_factor: float = 1,
):
    return LinearLR(
        optimizer,
        start_factor=start_factor,
        end_factor=end_factor,
        total_iters=num_warmup_steps,
    )


def fsdp2_clip_grad_norm_(parameters, max_norm, norm_type=2.0, error_if_nonfinite=False, foreach=None):
    """
    torch.nn.utils.clip_grad_norm_ can't run on cpu parameter DTensor.
    This function groups parameters by device mesh and clips grad norm for each group separately.
    """
    from collections import defaultdict

    from torch.nn.utils.clip_grad import _clip_grads_with_norm_, _get_total_norm

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    else:
        # prevent generators from being exhausted
        parameters = list(parameters)

    if not pgm.process_group_manager.enable_parallel:
        grads = [p.grad for p in parameters if p.grad is not None]
        total_norm = _get_total_norm(grads, norm_type, error_if_nonfinite, foreach)
        total_norm = total_norm.to(torch.cuda.current_device(), non_blocking=True)
        _clip_grads_with_norm_(parameters, max_norm, total_norm, foreach)
        return total_norm
    else:
        # Group parameters by device mesh
        mesh_groups = defaultdict(list)
        non_dtensor_params = []

        for p in parameters:
            if p.grad is not None:
                if isinstance(p.grad, DTensor):
                    # Use device mesh id as key to group parameters with same mesh
                    mesh_key = id(p.grad.device_mesh)
                    mesh_groups[mesh_key].append(p)
                else:
                    # Regular tensors (non-DTensor) go to separate group
                    non_dtensor_params.append(p)

        total_norms = []

        # Process each device mesh group separately
        for mesh_key, mesh_params in mesh_groups.items():
            grads = [p.grad for p in mesh_params]
            if grads:
                mesh_total_norm = _get_total_norm(grads, norm_type, error_if_nonfinite, foreach)
                mesh_total_norm = mesh_total_norm.to(torch.cuda.current_device(), non_blocking=True)
                _clip_grads_with_norm_(mesh_params, max_norm, mesh_total_norm, foreach)
                total_norms.append(mesh_total_norm)

        # Process non-DTensor parameters
        if non_dtensor_params:
            grads = [p.grad for p in non_dtensor_params]
            if grads:
                non_dtensor_total_norm = _get_total_norm(grads, norm_type, error_if_nonfinite, foreach)
                non_dtensor_total_norm = non_dtensor_total_norm.to(torch.cuda.current_device(), non_blocking=True)
                _clip_grads_with_norm_(non_dtensor_params, max_norm, non_dtensor_total_norm, foreach)
                total_norms.append(non_dtensor_total_norm)

        # Combine all norms - sum individual norm components then compute final norm
        if total_norms:
            if len(total_norms) == 1:
                return total_norms[0]
            else:
                # Sum the norm_type power of each norm, then take the final root
                # This avoids stacking tensors from different device meshes
                total_norm_sum = 0.0
                if norm_type == float("inf"):
                    # For infinity norm, take the maximum
                    max_norm = 0.0
                    for norm in total_norms:
                        max_norm = max(max_norm, norm.item())
                    return torch.tensor(max_norm, device=torch.cuda.current_device())
                else:
                    # For other norms, sum the powered norms then take the root
                    for norm in total_norms:
                        total_norm_sum += norm.item() ** norm_type
                    return torch.tensor(
                        total_norm_sum ** (1.0 / norm_type),
                        device=torch.cuda.current_device(),
                    )
        else:
            # No gradients found, return zero norm
            return torch.tensor(0.0, device=torch.cuda.current_device())
