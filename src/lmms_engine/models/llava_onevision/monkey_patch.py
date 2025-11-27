from functools import wraps

import torch
import torch.nn as nn
import transformers
from packaging import version
from transformers import PreTrainedModel

transformer_version = version.parse(transformers.__version__)
SUPPORTED_TRANSFORMER_VERSION = "4.46.1"
TRANSFORMER_DEPRECATION_WARNING = "Support for transformers versions < 4.46.1 will soon be discontinued due to issues with incorrect gradient accumulation. \n Please consider upgrading to avoid potential issues. See details: https://github.com/huggingface/transformers/pull/34191"

from loguru import logger

from lmms_engine.models.aero.monkey_patch import apply_liger_kernel_to_aero
from lmms_engine.models.monkey_patch import MONKEY_PATCHER


@MONKEY_PATCHER.register("llava_onevision", "liger")
def apply_liger_kernel_to_llava_onevision(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None,
    use_rmpad: bool = True,
):
    from transformers.models.llava_onevision.modeling_llava_onevision import (
        LlavaOnevisionForConditionalGeneration,
    )

    if fused_linear_cross_entropy:
        from .llava_ov_liger import forward as llava_ov_liger_forward

        if use_rmpad:

            def wrap_forward(func):
                @wraps(func)
                def wrapper(*args, **kwargs):
                    return func(use_rmpad=use_rmpad, *args, **kwargs)

                return wrapper

            llava_ov_liger_forward = wrap_forward(llava_ov_liger_forward)
        LlavaOnevisionForConditionalGeneration.forward = llava_ov_liger_forward

    apply_liger_kernel_to_aero(
        rope=rope,
        cross_entropy=cross_entropy,
        fused_linear_cross_entropy=fused_linear_cross_entropy,
        rms_norm=rms_norm,
        swiglu=swiglu,
        model=model.language_model,
        use_rmpad=use_rmpad,
    )


@MONKEY_PATCHER.register("llava_onevision", "video")
def apply_video_extensions_to_llava_onevision(
    model: PreTrainedModel = None,
    faster_token_stride: int = 10,
    mm_spatial_pool_stride: int = 2,
    mm_spatial_pool_mode: str = "bilinear",
):
    from transformers.models.llava_onevision.modeling_llava_onevision import (
        LlavaOnevisionModel,
    )

    if transformer_version < version.parse(SUPPORTED_TRANSFORMER_VERSION):
        logger.warning(TRANSFORMER_DEPRECATION_WARNING)

    # Store slow-fast parameters in model.config so forward can access them
    model.config.faster_token_stride = faster_token_stride
    model.config.mm_spatial_pool_stride = mm_spatial_pool_stride
    model.config.mm_spatial_pool_mode = mm_spatial_pool_mode

    # Initialize faster_token parameter for slow-fast frame processing
    # Applying video patch automatically enables slow-fast
    _initialize_faster_token(model)
    logger.info(f"Enabled slow-fast frame processing with stride={faster_token_stride}")

    # Apply custom forward for video processing
    # This forward supports both standard pooling and slow-fast frame modes
    from .llava_video_forward import forward as llava_video_model_forward

    LlavaOnevisionModel.forward = llava_video_model_forward
    logger.info(
        f"Applied video-aware forward to LlavaOnevisionModel "
        f"(pooling mode: {mm_spatial_pool_mode}, stride: {mm_spatial_pool_stride})"
    )

    logger.info("Successfully applied video extensions to LLaVA-OneVision model")


def _initialize_faster_token(model: PreTrainedModel):
    if hasattr(model.model, "faster_token"):
        logger.info("faster_token already initialized, skipping")
        return

    hidden_size = model.config.text_config.hidden_size
    dtype = model.dtype

    # Initialize faster_token with small random values
    embed_std = 1 / torch.sqrt(torch.tensor(hidden_size, dtype=torch.float32))
    faster_token = nn.Parameter(torch.randn(hidden_size, dtype=dtype) * embed_std)

    model.model.faster_token = faster_token
    logger.info(f"Initialized faster_token parameter with shape {faster_token.shape}")
