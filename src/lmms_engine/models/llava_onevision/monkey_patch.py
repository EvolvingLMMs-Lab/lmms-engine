from functools import wraps

import transformers
from packaging import version
from transformers import PreTrainedModel

transformer_version = version.parse(transformers.__version__)
SUPPORTED_TRANSFORMER_VERSION = "4.46.1"
TRANSFORMER_DEPRECATION_WARNING = "Support for transformers versions < 4.46.1 will soon be discontinued due to issues with incorrect gradient accumulation. \n Please consider upgrading to avoid potential issues. See details: https://github.com/huggingface/transformers/pull/34191"

from lmms_engine.models.aero.monkey_patch import apply_liger_kernel_to_aero
from lmms_engine.models.monkey_patch import MONKEY_PATCHER
from lmms_engine.utils.logging_utils import Logging


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
