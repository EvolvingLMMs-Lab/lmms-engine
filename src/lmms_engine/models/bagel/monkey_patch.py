import torch
from loguru import logger
from torch import nn

from lmms_engine.models.monkey_patch import MONKEY_PATCHER
from lmms_engine.utils import Logging

from .bagel import Bagel

try:
    from native_sparse_attention.module.native_sparse_attention import (
        COMPRESS_TYPE_TO_FUNC,
        COMPRESS_TYPE_TO_WEIGHT,
    )
except ImportError:
    logger.warning(
        "native_sparse_attention is not installed, please install with"
        " `uv pip install git+https://github.com/XunhaoLai/native-sparse-attention-triton.git`"
    )


def add_g_proj_to_attention_layers(model: Bagel, nsa_config: dict):
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
            compress_func = COMPRESS_TYPE_TO_FUNC[nsa_config["compress_type"]]
            compress_key = COMPRESS_TYPE_TO_WEIGHT[nsa_config["compress_type"]](
                attn_layer.config.num_key_value_heads,
                attn_layer.head_dim,
                nsa_config["kernel_size"],
            )
            compress_value = COMPRESS_TYPE_TO_WEIGHT[nsa_config["compress_type"]](
                attn_layer.config.num_key_value_heads,
                attn_layer.head_dim,
                nsa_config["kernel_size"],
            )
            intra_block_pe = torch.nn.Parameter(
                torch.zeros(
                    attn_layer.config.num_key_value_heads,
                    nsa_config["kernel_size"],
                    attn_layer.head_dim,
                )
            )
            attn_layer.compress_func = compress_func
            parameters = {
                "g_proj": g_proj,
                "compress_key": compress_key,
                "compress_value": compress_value,
                "intra_block_pe": intra_block_pe,
            }
            # set nsa config
            for key, value in nsa_config.items():
                setattr(attn_layer, key, value)
                setattr(attn_layer.config, key, value)

            for key, value in parameters.items():
                if isinstance(value, torch.nn.Module) or isinstance(
                    value, torch.nn.Parameter
                ):
                    value = value.to(dtype=model.dtype)
                if isinstance(value, torch.nn.Parameter):
                    attn_layer.register_parameter(key, value)
                elif isinstance(value, torch.Tensor):
                    attn_layer.register_parameter(
                        key, torch.nn.Parameter(value, requires_grad=True)
                    )
                else:
                    setattr(attn_layer, key, value)


@MONKEY_PATCHER.register("bagel", "nsa")
def apply_nsa_to_bagel(
    model: Bagel,
    block_size: int = 64,
    compress_type: str = "weightedpool",  # weightedpool, linear, avgpool
    kernel_size: int = 32,
    kernel_stride: int = 16,
    topk: int = 16,
    init_blocks: int = 1,
    local_blocks: int = 2,
    window_size: int = 512,
    **kwargs,
):
    """
    Apply NSA modifications to Bagel model.

    Args:
        model (Bagel): The Bagel model to modify
        **kwargs: Additional keyword arguments
    """
    nsa_config = {
        "block_size": block_size,
        "compress_type": compress_type,
        "kernel_size": kernel_size,
        "kernel_stride": kernel_stride,
        "topk": topk,
        "init_blocks": init_blocks,
        "local_blocks": local_blocks,
        "window_size": window_size,
    }
    Logging.info("Patch g_proj to bagel model")
    add_g_proj_to_attention_layers(model, nsa_config)
    Logging.info(
        f"NSA applied to bagel model, Model size: {sum(p.numel() for p in model.parameters()) / 1e9} B"
    )
    model.config.nsa_config = nsa_config

    from .nsa_op import forward_train as nsa_forward_train
    from .qwen2_navit import PackedAttentionMoT

    PackedAttentionMoT.forward_train = nsa_forward_train
