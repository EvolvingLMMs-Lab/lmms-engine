from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from transformers.modeling_outputs import BaseModelOutputWithPast


@dataclass
class BaseModelOutputWithPastAndRmpad(BaseModelOutputWithPast):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    seq_lens: Optional[torch.IntTensor] = None
    word_idx: Optional[torch.IntTensor] = None


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_unpad(q, k, cos, sin, position_ids):
    cos = (
        cos.squeeze().index_select(dim=0, index=position_ids.squeeze()).unsqueeze(1)
    )  # [total_bs_seq, 1, head_dim]
    sin = sin.squeeze().index_select(dim=0, index=position_ids.squeeze()).unsqueeze(1)

    # [total_bs_seq, head_num, head_dim] * [total_bs_seq, 1, head_dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# Copied from transformers.models.llama.modeling_llama._get_unpad_data
def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


# Copied from transformers.models.llama.modeling_llama._get_unpad_data
def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def _unpad_input(input_ids, attention_mask):
    valid_mask = attention_mask.squeeze(1).squeeze(1).eq(1)
    seqlens_in_batch = valid_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(valid_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(
        torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0)
    )
    input_ids = rearrange(input_ids, "b s ... -> (b s) ...")[indices]

    unpad_seq_len = input_ids.shape[0]

    return input_ids, indices, cu_seqlens, max_seqlen_in_batch
