import torch
import torch.nn.functional as F
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

try:
    from liger_kernel.transformers.functional import liger_cross_entropy
    from liger_kernel.transformers.fused_linear_cross_entropy import (
        LigerFusedLinearCrossEntropyLoss,
    )
    from liger_kernel.transformers.monkey_patch import (
        _patch_rms_norm_module,
        _patch_swiglu_module,
    )
    from liger_kernel.transformers.rms_norm import LigerRMSNorm
    from liger_kernel.transformers.rope import liger_rotary_pos_emb
    from liger_kernel.transformers.swiglu import LigerSwiGLUMLP
except ImportError:
    logger.warning("liger kernel not installed, please install it with `pip install liger-kernel`")


@MONKEY_PATCHER.register("bagel", "liger")
def apply_liger_kernel_to_bagel(
    rope: bool = True,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    model: Bagel | None = None,
) -> None:
    """
    Apply Liger kernels to replace original implementations in Bagel's Qwen2 backbone.
    """
    from . import qwen2_navit as bagel_qwen2_navit
    from .qwen2 import modeling_qwen2 as bagel_modeling_qwen2

    if rope:
        bagel_modeling_qwen2.apply_rotary_pos_emb = liger_rotary_pos_emb

        def _liger_rotary_pos_emb_packed(
            q: torch.Tensor,
            k: torch.Tensor,
            cos: torch.Tensor,
            sin: torch.Tensor,
            position_ids: torch.LongTensor | None = None,
            unsqueeze_dim: int = 1,
        ):
            # Packed attention stores tensors as (seq, heads, head_dim); adapt to 4D layout for Liger.
            if q.dim() == 3 and k.dim() == 3:
                q_shape = q.shape
                k_shape = k.shape

                q_4d = q.permute(1, 0, 2).unsqueeze(0).contiguous()
                k_4d = k.permute(1, 0, 2).unsqueeze(0).contiguous()
                cos_4d = cos.unsqueeze(0)
                sin_4d = sin.unsqueeze(0)

                q_rot, k_rot = liger_rotary_pos_emb(
                    q_4d,
                    k_4d,
                    cos_4d,
                    sin_4d,
                    position_ids=position_ids,
                    unsqueeze_dim=unsqueeze_dim,
                )

                q_rot = q_rot.squeeze(0).permute(1, 0, 2).contiguous()
                k_rot = k_rot.squeeze(0).permute(1, 0, 2).contiguous()

                assert q_rot.shape == q_shape, f"Unexpected shape after RoPE: {q_rot.shape} vs {q_shape}"
                assert k_rot.shape == k_shape, f"Unexpected shape after RoPE: {k_rot.shape} vs {k_shape}"
                return q_rot, k_rot

            return liger_rotary_pos_emb(q, k, cos, sin, position_ids=position_ids, unsqueeze_dim=unsqueeze_dim)

        bagel_qwen2_navit.apply_rotary_pos_emb = _liger_rotary_pos_emb_packed

    if rms_norm:
        bagel_modeling_qwen2.Qwen2RMSNorm = LigerRMSNorm
        bagel_qwen2_navit.Qwen2RMSNorm = LigerRMSNorm
    if swiglu:
        bagel_modeling_qwen2.Qwen2MLP = LigerSwiGLUMLP
        bagel_qwen2_navit.Qwen2MLP = LigerSwiGLUMLP

    if fused_linear_cross_entropy:
        original_ce_loss = Bagel.CrossEntropyLoss

        def liger_cross_entropy_loss(
            self,
            last_hidden_state,
            ce_loss_indexes,
            packed_label_ids,
            ce_loss_weights=None,
        ):
            if ce_loss_indexes is None or packed_label_ids is None:
                return None, torch.tensor(0, device=self.device)
            if self.config.ce_loss_reweighting or ce_loss_weights is not None:
                return original_ce_loss(
                    self,
                    last_hidden_state,
                    ce_loss_indexes,
                    packed_label_ids,
                    ce_loss_weights=ce_loss_weights,
                )
            hidden_states = last_hidden_state[ce_loss_indexes]
            if hidden_states.numel() == 0:
                return None, torch.tensor(0, device=self.device)
            if not hasattr(self, "_liger_fused_ce_module"):
                self._liger_fused_ce_module = LigerFusedLinearCrossEntropyLoss(reduction="mean")
            labels = (
                packed_label_ids
                if isinstance(packed_label_ids, torch.Tensor)
                else torch.tensor(packed_label_ids, device=hidden_states.device, dtype=torch.long)
            )
            labels = labels.to(device=hidden_states.device)
            loss = self._liger_fused_ce_module(self.language_model.lm_head.weight, hidden_states, labels)
            total_ce_tokens = self._count_ce_tokens(ce_loss_indexes)
            return loss, total_ce_tokens

        Bagel.CrossEntropyLoss = liger_cross_entropy_loss

    if model is not None:
        if not isinstance(model, Bagel):
            raise TypeError(f"Liger patch expected a Bagel model instance, got {type(model)}.")

        language_model = getattr(model, "language_model", None)
        if language_model is None or not hasattr(language_model, "model"):
            Logging.warning("Bagel model does not expose a Qwen2 backbone; skip instance level patch.")
            return

        qwen2_model = language_model.model

        if rms_norm:
            for module in qwen2_model.modules():
                if module.__class__.__name__ == "Qwen2RMSNorm":
                    _patch_rms_norm_module(module)
        if swiglu:
            for module in qwen2_model.modules():
                if module.__class__.__name__ == "Qwen2MLP":
                    _patch_swiglu_module(module, LigerSwiGLUMLP)

    Logging.info("Liger kernel applied to Bagel model.")


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
                if isinstance(value, torch.nn.Module) or isinstance(value, torch.nn.Parameter):
                    value = value.to(dtype=model.dtype)
                if isinstance(value, torch.nn.Parameter):
                    attn_layer.register_parameter(key, value)
                elif isinstance(value, torch.Tensor):
                    attn_layer.register_parameter(key, torch.nn.Parameter(value, requires_grad=True))
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
    Logging.info(f"NSA applied to bagel model, Model size: {sum(p.numel() for p in model.parameters()) / 1e9} B")
    model.config.nsa_config = nsa_config

    from .nsa_op import forward_train as nsa_forward_train
    from .qwen2_navit import PackedAttentionMoT

    PackedAttentionMoT.forward_train = nsa_forward_train
