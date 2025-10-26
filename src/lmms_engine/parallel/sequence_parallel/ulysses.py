# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Utilities for DeepSpeed Ulysses Sequence Parallelism.
DeepSpeed Ulysses Paper: https://arxiv.org/abs/2309.14509
Inspired from: https://github.com/deepspeedai/DeepSpeed/blob/master/deepspeed/sequence/layer.py
"""

from typing import Any, List, Optional

import torch
import torch.distributed as dist
from loguru import logger
from torch import Tensor
from torch.distributed import ProcessGroup

_ULYSSES_SEQUENCE_PARALLEL_GROUP = None


def set_ulysses_sequence_parallel_group(group: dist.ProcessGroup):
    """
    Set ulysses sequence parallel process group.
    """
    global _ULYSSES_SEQUENCE_PARALLEL_GROUP
    _ULYSSES_SEQUENCE_PARALLEL_GROUP = group


def get_ulysses_sequence_parallel_group() -> Optional[dist.ProcessGroup]:
    """
    Get ulysses sequence parallel process group.
    """
    global _ULYSSES_SEQUENCE_PARALLEL_GROUP
    return _ULYSSES_SEQUENCE_PARALLEL_GROUP


def get_ulysses_sequence_parallel_world_size(group: ProcessGroup = None) -> int:
    """
    Get ulysses sequence parallel world size.
    """
    group = get_ulysses_sequence_parallel_group() if group is None else group
    return dist.get_world_size(group) if group else 1


def get_ulysses_sequence_parallel_rank(group: ProcessGroup = None) -> int:
    """
    Get ulysses sequence parallel rank.
    """
    group = get_ulysses_sequence_parallel_group() if group is None else group
    return dist.get_rank(group) if group else 0


def patch_vlm_for_ulysses_input_slicing(model_class: type):
    """
    Applies a monkey patch to the forward method of a given model class
    to enable Ulysses sequence parallelism input slicing.
    """

    def _create_ulysses_wrapped_decoder_forward(original_forward):
        def ulysses_wrapped_decoder_forward(self, *args, **kwargs):
            inputs_embeds = kwargs.get("inputs_embeds")
            call_kwargs = kwargs.copy()

            current_ulysses_sp_size = get_ulysses_sequence_parallel_world_size()

            slice_now = (
                inputs_embeds is not None
                and current_ulysses_sp_size > 1
                and getattr(self, "_needs_initial_slice", True)
            )
            if slice_now:
                # [bs, seq_len, hidden_dim] slice at second dim
                # [total_seq_len, hidden_dim] slice at first dim
                slice_dim = 1 if inputs_embeds.dim() == 3 else 0
                call_kwargs["inputs_embeds"] = slice_input_tensor(inputs_embeds, dim=slice_dim, padding=True)
                self._needs_initial_slice = False
            try:
                return original_forward(self, *args, **call_kwargs)
            finally:
                if slice_now:
                    self._needs_initial_slice = True

        return ulysses_wrapped_decoder_forward

    original_forward = model_class.forward
    wrapped_forward = _create_ulysses_wrapped_decoder_forward(original_forward)
    model_class.forward = wrapped_forward
    logger.info(f"Monkey patch {model_class.__name__}.forward for Ulysses SP input slicing.")


def gather_seq_scatter_heads(
    x: Tensor,
    seq_dim: int,
    head_dim: int,
    unpadded_dim_size: int = 0,
    group: ProcessGroup = None,
) -> Tensor:
    """
    A func to sync embedding input with alltoall in sequence parallel
    gather sequence dimension and scatter head dim:
    e.g. seq_dim: 1, head_dim: 2
    [bsz, seq/n, h, ...] -> [bsz, seq, h/n, ...]
    """
    group = get_ulysses_sequence_parallel_group() if group is None else group
    if not group:
        return x
    sp_world = get_ulysses_sequence_parallel_world_size(group)
    x = SeqAllToAll.apply(group, x, head_dim, seq_dim)
    if unpadded_dim_size and unpadded_dim_size % sp_world != 0:
        padding_size = x.size(seq_dim) - unpadded_dim_size
        x = _unpad_tensor(x, seq_dim, padding_size)
    return x


def gather_heads_scatter_seq(x: Tensor, head_dim: int, seq_dim: int, group: ProcessGroup = None) -> Tensor:
    """
    A func to sync attention result with alltoall in sequence parallel
    gather head dimension and scatter seq dim:
    e.g. seq_dim: 1, head_dim: 2
    [bsz, seq, h/n, ...] -> [bsz, seq/n, h, ...]
    """
    group = get_ulysses_sequence_parallel_group() if group is None else group
    if not group:
        return x
    dim_size = x.size(seq_dim)
    sp_world = get_ulysses_sequence_parallel_world_size(group)
    if dim_size % sp_world != 0:
        padding_size = sp_world - (dim_size % sp_world)
        x = _pad_tensor(x, seq_dim, padding_size)
    return SeqAllToAll.apply(group, x, seq_dim, head_dim, False)


def _pad_tensor(x: Tensor, dim: int, padding_size: int) -> Tensor:
    shape = list(x.shape)
    shape[dim] = padding_size
    pad = torch.zeros(shape, dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=dim)


def _unpad_tensor(x: Tensor, dim: int, padding_size: int) -> Tensor:
    slc = [slice(None)] * len(x.shape)
    slc[dim] = slice(0, -padding_size)
    return x[slc]


def slice_input_tensor(x: Tensor, dim: int, padding: bool = True, group: ProcessGroup = None) -> Tensor:
    group = get_ulysses_sequence_parallel_group() if group is None else group
    sp_world_size = dist.get_world_size(group)
    sp_rank = get_ulysses_sequence_parallel_rank()
    dim_size = x.size(dim)
    # pad before slice
    if padding and dim_size % sp_world_size:
        padding_size = sp_world_size - (dim_size % sp_world_size)
        x = _pad_tensor(x, dim, padding_size)
    # slice the input tensor
    parts = x.size(dim) // sp_world_size
    slc = [slice(None)] * len(x.shape)
    slc[dim] = slice(sp_rank * parts, (sp_rank + 1) * parts)
    slc = tuple(slc)
    return x[slc].contiguous()


def all_to_all_tensor(
    local_input: Tensor,
    scatter_dim: int,
    gather_dim: int,
    group: Optional[dist.ProcessGroup] = None,
    async_op: bool = False,
):
    group = get_ulysses_sequence_parallel_group() if group is None else group
    seq_world_size = dist.get_world_size(group)
    input_list = [t.contiguous() for t in torch.tensor_split(local_input, seq_world_size, scatter_dim)]
    output_list = [torch.empty_like(input_list[0]) for _ in range(seq_world_size)]
    comm = dist.all_to_all(output_list, input_list, group=group, async_op=async_op)
    if async_op:

        def wait():
            comm.wait()
            return torch.cat(output_list, dim=gather_dim).contiguous()

        return wait
    return torch.cat(output_list, dim=gather_dim).contiguous()


def all_gather_tensor(
    local_tensor: Tensor,
    group: Optional[dist.ProcessGroup] = None,
    async_op: bool = False,
):
    group = get_ulysses_sequence_parallel_group() if group is None else group
    sp_world_size = dist.get_world_size(group=group)
    output_shape = list(local_tensor.shape)
    output_shape[0] = output_shape[0] * sp_world_size
    output = torch.empty(output_shape, dtype=local_tensor.dtype, device=local_tensor.device)
    dist.all_gather_into_tensor(output, local_tensor, group=group, async_op=async_op)
    return output


class SeqAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        local_input: Tensor,
        scatter_dim: int,
        gather_dim: int,
        async_op: bool = False,
    ) -> Tensor:
        ctx.group = group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.async_op = async_op
        return all_to_all_tensor(local_input, scatter_dim, gather_dim, group, async_op)

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> tuple[None, Tensor, None, None]:
        input_t = torch.cat(grad_output[1:], dim=ctx.gather_dim).contiguous() if ctx.async_op else grad_output[0]
        return (
            None,
            all_to_all_tensor(input_t, ctx.gather_dim, ctx.scatter_dim, ctx.group, False),
            None,
            None,
            None,
            None,
        )


class Gather(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        local_tensor: Tensor,
        gather_dim: int,
        grad_scaler: bool = True,
        async_op=False,
    ) -> Tensor:
        ctx.group = group
        ctx.gather_dim = gather_dim
        ctx.grad_scaler = grad_scaler
        ctx.async_op = async_op

        sp_world_size = dist.get_world_size(group=group)
        ctx.sp_world_size = sp_world_size

        sp_rank = dist.get_rank(group=group)
        ctx.sp_rank = sp_rank

        local_shape = list(local_tensor.size())
        split_size = local_shape[0]
        part_size = local_shape[gather_dim]  # store original size
        ctx.part_size = part_size

        output = all_gather_tensor(local_tensor, group, async_op)
        return torch.cat(output.split(split_size, dim=0), dim=gather_dim)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Any:
        if ctx.grad_scaler:
            grad_output = grad_output * ctx.sp_world_size
        return (
            None,
            grad_output.split(ctx.part_size, dim=ctx.gather_dim)[ctx.sp_rank].contiguous(),
            None,
            None,
            None,
            None,
        )


def gather_outpus_and_unpad(*args, **kwargs):
    raise RuntimeError(
        "please use verl.utils.ulysses.gather_outputs_and_unpad instead of verl.utils.ulysses.gather_outpus_and_unpad"
    )


def gather_outputs_and_unpad(
    x: Tensor,
    gather_dim: int,
    unpad_dim: int = None,
    padding_size: int = 0,
    grad_scaler: bool = True,
    group: Optional[dist.ProcessGroup] = None,
):
    """
    Gather a tensor across a process group and optionally unpad its padded elements.

    Args:
        x (Tensor): Input tensor to gather.
        gather_dim (int): Dimension along which to gather across ranks.
        unpad_dim (int, optional): Dimension from which to remove padding. If None, no unpadding.
        padding_size (int): Number of padding elements to remove on `unpad_dim`. Defaults to 0.
        grad_scaler (bool): Whether to apply gradient scaling during gather. Defaults to True.
        group (ProcessGroup, optional): Process group for gathering. If None, uses
            `get_ulysses_sequence_parallel_group()`. If still None, returns `x` unchanged.

    Returns:
        Tensor: The gathered tensor, with padding removed if requested.
    """
    group = get_ulysses_sequence_parallel_group() if group is None else group
    if group is None:
        return x
    x = Gather.apply(group, x, gather_dim, grad_scaler)
    if unpad_dim is not None:
        assert isinstance(padding_size, int), "padding size is not given or is not an integer"
        if padding_size == 0:
            return x
        x = _unpad_tensor(x, unpad_dim, padding_size)
    return x


def ulysses_pad(
    input_ids_rmpad: torch.Tensor,
    position_ids_rmpad: Optional[torch.Tensor] = None,
    sp_size: int = 1,
):
    if position_ids_rmpad is not None:
        assert position_ids_rmpad.size(-2) == 1
        assert input_ids_rmpad.size(-1) == position_ids_rmpad.size(-1)
    if sp_size <= 1:
        return input_ids_rmpad, position_ids_rmpad, 0
    _, total_seq_len = input_ids_rmpad.shape
    pad_size = (sp_size - total_seq_len % sp_size) % sp_size
    if pad_size > 0:
        input_ids_rmpad = torch.nn.functional.pad(input_ids_rmpad, (0, pad_size), value=0)
        if position_ids_rmpad is not None:
            pad_pos_ids = torch.arange(pad_size, device=position_ids_rmpad.device).unsqueeze(0)
            if position_ids_rmpad.dim() == 3:
                pad_pos_ids = pad_pos_ids.unsqueeze(0).repeat(3, 1, 1)
            position_ids_rmpad = torch.cat((position_ids_rmpad, pad_pos_ids), dim=-1)
    return input_ids_rmpad, position_ids_rmpad, pad_size


def ulysses_pad_and_slice_inputs(
    input_ids_rmpad: torch.Tensor,
    position_ids_rmpad: Optional[torch.Tensor] = None,
    sp_size: int = 1,
):
    """
    Pad and slice input_ids to be divisible by sp_size
    Pad position_ids to be divisible by sp_size.

    Note both input_ids_rmpad and position_ids_rmpad will be padded and sliced.

    The is the utility of pre-forward for ulysses sequence parallelism

    Args:
        input_ids_rmpad: shape of [bsz, seqlen]
        position_ids_rmpad: shape of [bsz, seqlen], where bsz must be 1
        sp_size (int): ulysses sequence parallelism size

    Returns:
        torch.Tensor: padded and sliced input_ids
        torch.Tensor: padded and sliced position_ids
        int: pad size
    """
    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(input_ids_rmpad, position_ids_rmpad, sp_size)
    input_ids_rmpad = slice_input_tensor(input_ids_rmpad, dim=1, padding=False)
    if position_ids_rmpad is not None:
        position_ids_rmpad = slice_input_tensor(position_ids_rmpad, dim=1, padding=False)
    return input_ids_rmpad, position_ids_rmpad, pad_size


def validate_ulysses_config(num_heads, ulysses_sequence_size):
    if ulysses_sequence_size > 1:
        assert (
            num_heads % ulysses_sequence_size == 0
        ), f"num_heads ({num_heads}) must be divisible by ulysses sequence size({ulysses_sequence_size})"


def calculate_seq_len_per_rank(seq_len: List[int]):
    # The seq len is the largest val
    total_seq_len = max(seq_len)
    sp_size = get_ulysses_sequence_parallel_world_size()
    pad_size = (sp_size - total_seq_len % sp_size) % sp_size
    per_seq_len = (total_seq_len + pad_size) // sp_size
    start_from = per_seq_len * get_ulysses_sequence_parallel_rank()
    end_at = start_from + per_seq_len
    cur_seq_len = []
    for sl in seq_len:
        if sl <= start_from and len(cur_seq_len) == 0:
            cur_seq_len.append(0)
        elif sl <= end_at and sl > start_from:
            cur_seq_len.append(sl - start_from)
    # If the last seq len is not equal to per_seq_len, we need to add it
    if cur_seq_len[-1] != per_seq_len:
        cur_seq_len.append(per_seq_len)
    return cur_seq_len


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=2, repeats=n_rep). The hidden states go from (batch,
    seqlen, num_key_value_heads, head_dim) to (batch, seqlen, num_attention_heads, head_dim)
    """
    slen, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :].expand(slen, num_key_value_heads, n_rep, head_dim)
    return hidden_states.reshape(slen, num_key_value_heads * n_rep, head_dim)


def pad_and_mask_visual_for_ulysses(
    mask: Tensor,
    sp_size: int = 1,
    group: ProcessGroup = None,
) -> Tensor:
    """
    Pad and apply rank-specific masking for visual masks (image/video) in Ulysses sequence parallelism.

    This function prepares visual masks for distribution across sequence parallel ranks by:
    1. Padding the mask to be divisible by sp_size along dimension 0
    2. Masking out chunks that don't belong to the current rank (set to False)

    Args:
        mask (Tensor): Visual mask tensor of shape [seq_len, ...] where seq_len is the sequence length.
                       Typically has shape [seq_len] or [seq_len, hidden_dim].
        sp_size (int): Sequence parallel size. If <= 1, returns mask unchanged. Default: 1.
        group (ProcessGroup, optional): Process group for rank information. If None, uses the
                                       default Ulysses sequence parallel group.

    Returns:
        Tensor: Padded mask with shape [padded_seq_len, ...] where padded_seq_len is the smallest
                multiple of sp_size >= seq_len. Only the chunk belonging to the current rank
                contains the original mask values; other chunks are zeros.

    Examples:
        Example 1 - Basic usage with sp_size=2:
            Input mask: torch.tensor([True, True, False])  # shape: [3]
            sp_size: 2
            Result on rank 0: torch.tensor([True, True, False, False])  # shape: [4], keeps positions [0:2]
            Result on rank 1: torch.tensor([False, False, False, False])  # shape: [4], keeps positions [2:4]

        Example 2 - Already divisible sequence:
            Input mask: torch.tensor([True, True, True, True])  # shape: [4]
            sp_size: 2
            Result on rank 0: torch.tensor([True, True, False, False])  # positions [0:2]
            Result on rank 1: torch.tensor([False, False, True, True])  # positions [2:4]

        Example 3 - Higher dimensional mask:
            Input mask: torch.tensor([[True], [False], [True]])  # shape: [3, 1]
            sp_size: 2
            Result on rank 0: torch.tensor([[True], [False], [False], [False]])  # shape: [4, 1]

    Edge Cases:
        - Empty sequence (seq_len=0): Returns padded zeros of shape [sp_size, ...]
        - sp_size=1: Returns original mask unchanged (no parallelism)
        - seq_len < sp_size: Pads to sp_size, some ranks may get only padding

    Performance Notes:
        - Padding is done on device using torch.nn.functional.pad (no CPU sync)
        - Memory overhead is minimal: at most (sp_size - 1) additional elements
    """
    if sp_size <= 1:
        return mask

    assert sp_size > 0, f"sp_size must be positive, got {sp_size}"
    assert mask is not None, "mask cannot be None for sequence parallelism"

    group = get_ulysses_sequence_parallel_group() if group is None else group
    if group is None:
        return mask

    # Get rank information
    sp_rank = get_ulysses_sequence_parallel_rank(group)

    # Get the original sequence length
    original_seq_len = mask.size(0)

    # Calculate padding needed to make divisible by sp_size
    # Calculate padding needed: (sp_size - remainder) mod sp_size handles both divisible (0) and non-divisible cases
    pad_size = (sp_size - original_seq_len % sp_size) % sp_size

    # Pad the mask if necessary
    if pad_size > 0:
        padding = [0] * (2 * mask.ndim)
        padding[0] = pad_size  # Pad at the end of dimension 0
        mask = torch.nn.functional.pad(mask.float(), padding, value=0.0)

    # Calculate chunk size per rank
    padded_seq_len = mask.size(0)
    chunk_size = padded_seq_len // sp_size

    # Calculate which chunk indices belong to this rank
    rank_start = sp_rank * chunk_size
    rank_end = (sp_rank + 1) * chunk_size

    # Create a mask for the current rank
    # Set all elements to False, then set only the current rank's chunks to their original values
    rank_mask = torch.zeros_like(mask)
    rank_mask[rank_start:rank_end] = mask[rank_start:rank_end]

    return rank_mask.bool() if mask.dtype == torch.bool else rank_mask


def get_visual_embeds_for_rank(
    visual_embeds: Tensor,
    original_mask: Tensor,
    sp_size: int,
    group: ProcessGroup = None,
) -> List[Tensor]:
    """
    Distribute visual embeddings to the current rank based on sequence parallel split.

    This function solves the challenge of distributing visual embeddings when using sequence
    parallelism: the embeddings are stored densely (only for True mask positions), but the
    sequence is split spatially across ranks. Each rank receives only the embeddings that
    correspond to True mask values within its assigned sequence chunk.

    The distribution logic:
    1. Determine which positions [rank_start:rank_end] belong to this rank in the padded sequence
    2. Count how many True values appear before rank_start in the original mask
    3. Count how many True values appear in [rank_start:rank_end] in the original mask
    4. Return the corresponding slice of visual_embeds

    Args:
        visual_embeds (Tensor): Dense visual embeddings of shape [num_true_values, hidden_dim],
                               containing embeddings only for positions where original_mask is True.
        original_mask (Tensor): Boolean mask of shape [seq_len] indicating which positions have
                               visual features. Must be the unpadded, original mask before any
                               sequence parallel transformations.
        sp_size (int): Sequence parallel size. Number of ranks across which to distribute.
                      If <= 1, returns all embeddings unchanged.
        group (ProcessGroup, optional): Process group for rank information. If None, uses the
                                       default Ulysses sequence parallel group.

    Returns:
        Tensor: Subset of visual_embeds for the current rank, shape [num_true_in_chunk, hidden_dim].
               Returns empty tensor if the current rank's chunk contains no True values.

    Examples:
        Example 1 - Basic distribution:
            original_mask: torch.tensor([True, True, True, False])  # 3 visual features
            visual_embeds: torch.randn(3, 768)  # embeddings for the 3 True positions
            sp_size: 2

            After padding mask to [4]:
            - rank0 gets positions [0:2] → mask[0:2] = [True, True] → returns embeds[0:2]
            - rank1 gets positions [2:4] → mask[2:4] = [True, False] → returns embeds[2:3]

        Example 2 - Uneven distribution:
            original_mask: torch.tensor([True, False, False, True, True])  # 3 visual features
            visual_embeds: torch.randn(3, 768)
            sp_size: 2

            After padding mask to [6]:
            - rank0 gets positions [0:3] → mask[0:3] = [True, False, False] → returns embeds[0:1]
            - rank1 gets positions [3:6] → mask[3:6] = [True, True, <pad>] → returns embeds[1:3]

        Example 3 - Rank gets no visual features:
            original_mask: torch.tensor([True, True, False])
            visual_embeds: torch.randn(2, 768)
            sp_size: 2

            After padding mask to [4]:
            - rank0 gets positions [0:2] → mask[0:2] = [True, True] → returns embeds[0:2]
            - rank1 gets positions [2:4] → mask[2:4] = [False, <pad>] → returns empty tensor []

    Edge Cases:
        - sp_size=1: Returns all embeddings unchanged (no parallelism)
        - Empty embeddings: Returns empty tensor
        - All False mask: Returns empty tensor for all ranks
        - Rank entirely in padding region: Returns empty list

    Performance Notes:
        - Uses tensor slicing and sum operations on mask (stays on device)
        - No data movement between ranks; purely local computation
        - Memory efficient: only stores embeddings for current rank
    """
    if sp_size <= 1:
        return visual_embeds

    assert sp_size > 0, f"sp_size must be positive, got {sp_size}"
    assert visual_embeds is not None, "visual_embeds cannot be None"
    assert original_mask is not None, "original_mask cannot be None"

    group = get_ulysses_sequence_parallel_group() if group is None else group
    if group is None:
        return visual_embeds

    sp_rank = get_ulysses_sequence_parallel_rank(group)
    original_seq_len = original_mask.size(0)

    # Calculate chunk size in padded sequence
    pad_size = (sp_size - original_seq_len % sp_size) % sp_size
    padded_seq_len = original_seq_len + pad_size
    chunk_size = padded_seq_len // sp_size

    # Determine which positions in original sequence belong to this rank
    rank_start = sp_rank * chunk_size
    rank_end = (sp_rank + 1) * chunk_size

    # Clamp to original sequence length (padding positions have no real values)
    rank_start_in_orig = min(rank_start, original_seq_len)
    rank_end_in_orig = min(rank_end, original_seq_len)

    # If this rank is entirely in the padding region, return empty tensor
    if rank_start_in_orig >= original_seq_len:
        return torch.empty(
            0,
            *visual_embeds.shape[1:],
            device=visual_embeds.device,
            dtype=visual_embeds.dtype,
        )

    # Use cumsum to compute indices on GPU, avoiding intermediate .item() calls
    # cumsum gives us the count of True values up to each position
    cumsum_mask = torch.cumsum(original_mask.int(), dim=0)

    # Count of True values before this rank's chunk (exclusive)
    count_before = (
        cumsum_mask[rank_start_in_orig - 1]
        if rank_start_in_orig > 0
        else torch.tensor(0, device=cumsum_mask.device, dtype=cumsum_mask.dtype)
    )

    # Count of True values up to the end of this rank's chunk (inclusive)
    count_up_to_end = (
        cumsum_mask[rank_end_in_orig - 1]
        if rank_end_in_orig > 0
        else torch.tensor(0, device=cumsum_mask.device, dtype=cumsum_mask.dtype)
    )

    # Tensor slicing will implicitly call .item() on the indices, but the computation
    # stayed on GPU. This is more efficient than calling .sum().item() multiple times.
    return visual_embeds[count_before:count_up_to_end]
