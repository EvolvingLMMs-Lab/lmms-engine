import torch
from torch import nn

from lmms_engine.models.monkey_patch import MONKEY_PATCHER
from lmms_engine.utils import Logging

from .bagel import Bagel


def add_g_proj_to_attention_layers(
    model: Bagel, block_size: int, block_counts: int, window_size: int
):
    """
    Add g_proj linear layers to all attention layers in the Bagel model.

    Args:
        model (Bagel): The Bagel model to modify
    """
    # Access the language model's decoder layers
    for layer in model.language_model.model.layers:
        # Each layer has a self_attn module
        if hasattr(layer, "self_attn"):
            attn_layer = layer.self_attn
            g_proj = nn.Linear(model.hidden_size, model.num_heads * 3, bias=False)
            g_proj = g_proj.to(model.dtype)
            # Add g_proj linear layer with size (hidden_size, num_heads * 3)
            attn_layer.g_proj = g_proj
            attn_layer.block_size = block_size
            attn_layer.window_size = window_size
            attn_layer.block_counts = block_counts
            setattr(attn_layer.config, "block_size", block_size)
            setattr(attn_layer.config, "window_size", window_size)
            setattr(attn_layer.config, "block_counts", block_counts)
            # Initialize the g_proj layer (optional - you may want different initialization)
            nn.init.normal_(attn_layer.g_proj.weight, std=0.02)


@MONKEY_PATCHER.register("bagel", "nsa")
def apply_nsa_to_bagel(
    model: Bagel, block_size: int, block_counts: int, window_size: int, **kwargs
):
    """
    Apply NSA (Neural Sparse Attention) modifications to Bagel model.

    Args:
        model (Bagel): The Bagel model to modify
        **kwargs: Additional keyword arguments
    """
    Logging.info("Patch g_proj to bagel model")
    add_g_proj_to_attention_layers(model, block_size, block_counts, window_size)
    Logging.info(
        f"g_proj patched to bagel model, Model size: {sum(p.numel() for p in model.parameters()) / 1e9} B"
    )

    from .nsa_op import forward_train as nsa_forward_train
    from .qwen2_navit import PackedAttentionMoT

    PackedAttentionMoT.forward_train = nsa_forward_train
