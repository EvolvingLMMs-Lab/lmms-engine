# Most of the code copied from https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/transformers/monkey_patch.py
# Modified to work on patch our models

import inspect
from functools import partial, wraps
from typing import Callable

from packaging import version

try:
    from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss
    from liger_kernel.transformers.functional import liger_cross_entropy
    from liger_kernel.transformers.geglu import LigerGEGLUMLP
    from liger_kernel.transformers.layer_norm import LigerLayerNorm
    from liger_kernel.transformers.model.qwen2 import (
        lce_forward_deprecated as qwen2_lce_forward_deprecated,
    )
    from liger_kernel.transformers.qwen2vl_mrope import liger_multimodal_rotary_pos_emb
    from liger_kernel.transformers.rms_norm import LigerRMSNorm
    from liger_kernel.transformers.rope import liger_rotary_pos_emb
    from liger_kernel.transformers.swiglu import LigerSwiGLUMLP
except:
    print(
        "liger kernel not installed, please install it with `pip install liger-kernel`"
    )

import transformers
from transformers import (
    PreTrainedModel,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLModel,
    Qwen2_5_VLTextModel,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionTransformerPretrainedModel,
)

transformer_version = version.parse(transformers.__version__)
SUPPORTED_TRANSFORMER_VERSION = "4.46.1"
TRANSFORMER_DEPRECATION_WARNING = "Support for transformers versions < 4.46.1 will soon be discontinued due to issues with incorrect gradient accumulation. \n Please consider upgrading to avoid potential issues. See details: https://github.com/huggingface/transformers/pull/34191"

from ...utils.logging_utils import Logging

try:
    import peft

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


def _bind_method_to_module(module, method_name: str, new_method: Callable):
    # Binds a new method to a module instance so that self is passed as the first argument
    module.__dict__[method_name] = new_method.__get__(module, module.__class__)


def _patch_rms_norm_module(
    module, offset=0.0, eps=1e-6, casting_mode="llama", in_place=True, row_mode=None
):
    # Check if the module is a PEFT ModulesToSaveWrapper
    # If it is, we need to patch the modules_to_save.default and original_modules
    if PEFT_AVAILABLE and isinstance(module, peft.utils.other.ModulesToSaveWrapper):
        module.modules_to_save.default.offset = offset
        module.modules_to_save.default.casting_mode = casting_mode
        module.modules_to_save.default.variance_epsilon = (
            getattr(module, "variance_epsilon", None)
            or getattr(module, "eps", None)
            or eps
        )
        module.modules_to_save.default.in_place = in_place
        module.modules_to_save.default.row_mode = row_mode
        module.original_module.offset = offset
        module.original_module.casting_mode = casting_mode
        module.original_module.variance_epsilon = (
            getattr(module, "variance_epsilon", None)
            or getattr(module, "eps", None)
            or eps
        )
        module.original_module.in_place = in_place
        module.original_module.row_mode = row_mode
        _bind_method_to_module(
            module.modules_to_save.default, "forward", LigerRMSNorm.forward
        )
        _bind_method_to_module(
            module.modules_to_save.default, "extra_repr", LigerRMSNorm.extra_repr
        )
        _bind_method_to_module(module.original_module, "forward", LigerRMSNorm.forward)
        _bind_method_to_module(
            module.original_module, "extra_repr", LigerRMSNorm.extra_repr
        )
        module.modules_to_save.default.__class__.__name__ = LigerRMSNorm.__name__
        module.original_module.__class__.__name__ = LigerRMSNorm.__name__
    else:
        module.offset = offset
        module.casting_mode = casting_mode
        module.variance_epsilon = (
            getattr(module, "variance_epsilon", None)
            or getattr(module, "eps", None)
            or eps
        )
        module.in_place = in_place
        module.row_mode = row_mode
        _bind_method_to_module(module, "forward", LigerRMSNorm.forward)
        _bind_method_to_module(module, "extra_repr", LigerRMSNorm.extra_repr)
        module.__class__.__name__ = LigerRMSNorm.__name__


def _patch_layer_norm_module(module, eps=1e-6):
    module.variance_epsilon = (
        getattr(module, "variance_epsilon", None) or getattr(module, "eps", None) or eps
    )
    module.hidden_size = module.normalized_shape
    _bind_method_to_module(module, "forward", LigerLayerNorm.forward)
    _bind_method_to_module(module, "extra_repr", LigerLayerNorm.extra_repr)
    module.__class__.__name__ = LigerLayerNorm.__name__


def _patch_swiglu_module(module, liger_module):
    _bind_method_to_module(module, "forward", liger_module.forward)
    module.__class__.__name__ = liger_module.__name__


def _patch_geglu_module(module):
    _bind_method_to_module(module, "forward", LigerGEGLUMLP.forward)
    module.__class__.__name__ = LigerGEGLUMLP.__name__


def apply_liger_kernel_to_qwen2_5_vl(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None,
    use_rmpad: bool = False,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace Qwen2-VL models.
    NOTE: Qwen2.5-VL is not available in transformers<4.48.2
    Args:
        cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
        fused_linear_cross_entropy (bool):
            Whether to apply Liger's fused linear cross entropy loss. Default is True.
            `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
            If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
        rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
        swiglu (bool): Whether to apply Liger's SwiGLU MLP. Default is True.
        model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
        loaded. Default is None.
    """
    assert not (
        cross_entropy and fused_linear_cross_entropy
    ), "cross_entropy and fused_linear_cross_entropy cannot both be True."

    from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel

    from .qwen2_5_vl_liger import lce_forward as qwen2_5_vl_lce_forward

    if use_rmpad:

        def wrap_forward(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(use_rmpad=use_rmpad, *args, **kwargs)

            return wrapper

        qwen2_5_vl_lce_forward = wrap_forward(qwen2_5_vl_lce_forward)

    if rope:
        modeling_qwen2_5_vl.apply_multimodal_rotary_pos_emb = (
            liger_multimodal_rotary_pos_emb
        )
    if rms_norm:
        modeling_qwen2_5_vl.Qwen2RMSNorm = LigerRMSNorm
    if cross_entropy:
        modeling_qwen2_5_vl.CrossEntropyLoss = LigerCrossEntropyLoss
    if fused_linear_cross_entropy:
        modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = (
            qwen2_5_vl_lce_forward
        )
    if swiglu:
        modeling_qwen2_5_vl.Qwen2MLP = LigerSwiGLUMLP

    if use_rmpad:
        from .rmpad.qwen2_5_vl_ops import attn_forward as qwen2_ops_attn_forward
        from .rmpad.qwen2_5_vl_ops import (
            decoder_layer_forward as qwen2_ops_decoder_layer_forward,
        )
        from .rmpad.qwen2_5_vl_ops import (
            text_model_forward as qwen2_ops_text_model_forward,
        )
        from .rmpad.qwen2_5_vl_ops import vl_model_forward as qwen2_ops_vl_model_forward

        modeling_qwen2_5_vl.Qwen2_5_VLModel.forward = qwen2_ops_vl_model_forward
        modeling_qwen2_5_vl.Qwen2_5_VLTextModel.forward = qwen2_ops_text_model_forward
        modeling_qwen2_5_vl.Qwen2_5_VLDecoderLayer.forward = (
            qwen2_ops_decoder_layer_forward
        )
        modeling_qwen2_5_vl.Qwen2_5_VLAttention.forward = qwen2_ops_attn_forward
    apply_liger_kernel_to_qwen2_audio(use_rmpad=use_rmpad)

    if model is not None:
        # The model instance already exists, so we need to additionally patch the
        # instance variables that reference already-instantiated modules
        if isinstance(model, Qwen2_5_VLForConditionalGeneration):
            text_model: Qwen2_5_VLTextModel = model.model.language_model
            vision_model: Qwen2_5_VisionTransformerPretrainedModel = model.model.visual
        elif isinstance(model, Qwen2_5_VLModel):
            # Note: language_model and visual properties can be accessed throught conditional class for BC.
            # Not sure if it is subject to changes in the future.
            # Reference: https://github.com/huggingface/transformers/blob/v4.52.4/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L1823
            text_model: Qwen2_5_VLTextModel = model.language_model
            vision_model: Qwen2_5_VisionTransformerPretrainedModel = model.visual
        elif isinstance(model, Qwen2_5_VLTextModel):
            text_model: Qwen2_5_VLTextModel = model
            vision_model = None
        else:
            # Note: Currently there's no support for patching vision model only. Feel free to raise an issue if needed.
            raise TypeError(
                f"Unsupported Qwen2VL model type. `model` must be `Qwen2VLForConditionalGeneration`, `Qwen2VLModel` or `Qwen2VLTextModel`. Got: {type(model)}"
            )

        if vision_model is not None:
            # Patch Qwen2_5_VisionTransformerPretrainedModel
            for vision_block in model.visual.blocks:
                if rms_norm:
                    _patch_rms_norm_module(vision_block.norm1)
                    _patch_rms_norm_module(vision_block.norm2)

        if text_model is not None:
            if rms_norm:
                _patch_rms_norm_module(text_model.norm)
            for decoder_layer in text_model.layers:
                if swiglu:
                    _patch_swiglu_module(decoder_layer.mlp, LigerSwiGLUMLP)
                if rms_norm:
                    _patch_rms_norm_module(decoder_layer.input_layernorm)
                    _patch_rms_norm_module(decoder_layer.post_attention_layernorm)


def apply_liger_kernel_to_aero(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None,
    use_rmpad: bool = False,
) -> None:
    assert not (
        cross_entropy and fused_linear_cross_entropy
    ), "cross_entropy and fused_linear_cross_entropy cannot both be True."

    from transformers.models.qwen2 import modeling_qwen2
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Model

    from ..aero import modeling_aero

    if rope:
        modeling_qwen2.apply_rotary_pos_emb = liger_rotary_pos_emb
    if rms_norm:
        modeling_qwen2.Qwen2RMSNorm = LigerRMSNorm

    if cross_entropy:
        if transformer_version >= version.parse(SUPPORTED_TRANSFORMER_VERSION):
            from transformers.loss.loss_utils import nn

            nn.functional.cross_entropy = liger_cross_entropy
        else:
            Logging.warning(TRANSFORMER_DEPRECATION_WARNING)
            modeling_qwen2.CrossEntropyLoss = LigerCrossEntropyLoss

    if fused_linear_cross_entropy:
        from .qwen2_liger import qwen2_lce_forward

        if use_rmpad:

            def wrap_forward(func):
                @wraps(func)
                def wrapper(*args, **kwargs):
                    return func(use_rmpad=use_rmpad, *args, **kwargs)

                return wrapper

            qwen2_lce_forward = wrap_forward(qwen2_lce_forward)
        modeling_qwen2.Qwen2ForCausalLM.forward = qwen2_lce_forward

    if swiglu:
        modeling_qwen2.Qwen2MLP = LigerSwiGLUMLP
    apply_liger_kernel_to_qwen2_audio(use_rmpad=use_rmpad)

    if use_rmpad:
        from .rmpad.aero_ops import forward as aero_ops_forward
        from .rmpad.qwen2_ops import attn_forward as qwen2_ops_attn_forward
        from .rmpad.qwen2_ops import (
            decoder_layer_forward as qwen2_ops_decoder_layer_forward,
        )
        from .rmpad.qwen2_ops import model_forward as qwen2_ops_model_forward

        modeling_qwen2.Qwen2Model.forward = qwen2_ops_model_forward
        modeling_qwen2.Qwen2DecoderLayer.forward = qwen2_ops_decoder_layer_forward
        modeling_qwen2.Qwen2Attention.forward = qwen2_ops_attn_forward
        modeling_aero.AeroForConditionalGeneration.forward = aero_ops_forward

    if model is not None:
        # The model instance already exists, so we need to additionally patch the
        # instance variables that reference already-instantiated modules

        # get the base model from the model instance
        base_model: Qwen2Model = getattr(
            model.language_model,
            model.language_model.base_model_prefix,
            model.language_model,
        )

        if rms_norm:
            _patch_rms_norm_module(base_model.norm)

        for decoder_layer in base_model.layers:
            if swiglu:
                _patch_swiglu_module(decoder_layer.mlp, LigerSwiGLUMLP)
            if rms_norm:
                _patch_rms_norm_module(decoder_layer.input_layernorm)
                _patch_rms_norm_module(decoder_layer.post_attention_layernorm)


def apply_liger_kernel_to_aero_omni(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None,
    use_rmpad: bool = False,
):
    apply_liger_kernel_to_aero(
        rope=rope,
        cross_entropy=cross_entropy,
        fused_linear_cross_entropy=fused_linear_cross_entropy,
        rms_norm=rms_norm,
        swiglu=swiglu,
        model=model,
        use_rmpad=use_rmpad,
    )

    if use_rmpad:
        from ..aero_omni import modeling_aero_omni
        from .rmpad.aero_omni_ops import forward as aero_omni_ops_forward

        modeling_aero_omni.AeroOmniForConditionalGeneration.forward = (
            aero_omni_ops_forward
        )


def apply_liger_kernel_to_qwen2_audio(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None,
    use_rmpad: bool = True,
):
    from transformers import Qwen2AudioEncoder
    from transformers.models.qwen2_audio.modeling_qwen2_audio import (
        Qwen2AudioAttention,
        Qwen2AudioEncoderLayer,
    )

    if use_rmpad:
        from .rmpad.qwen2_audio_ops import (
            encoder_forward as qwen2_audio_encoder_forward,
        )
        from .rmpad.qwen2_audio_ops import (
            encoder_layer_forward as qwen2_audio_encoder_layer_forward,
        )
        from .rmpad.qwen2_audio_ops import (
            flash_attn_forward as qwen2_audio_flash_attn_forward,
        )

        Qwen2AudioEncoder.forward = qwen2_audio_encoder_forward
        Qwen2AudioEncoderLayer.forward = qwen2_audio_encoder_layer_forward
        Qwen2AudioAttention.forward = qwen2_audio_flash_attn_forward


CUSTOM_MODEL_TYPE_TO_APPLY_LIGER_FN = {
    "aero": apply_liger_kernel_to_aero,
    "aero_omni": apply_liger_kernel_to_aero_omni,
    "qwen2_5_vl": apply_liger_kernel_to_qwen2_5_vl,
}


def _apply_liger_kernel(model_type: str, **kwargs) -> None:
    """
    Applies Liger kernels based on the specified model type. The custom
    kernels for the specified model type will be applied with the provided
    keyword arguments, otherwise the default configuration will be used.

    ** Note: Calling _apply_liger_kernel() after model initialization
    will not be able to fully patch models. This must be called before model initialization.
    If the model has already been instantiated

    Args:
        - model_type: the model types as defined in transformers/models/auto/modeling_auto.py
          and specified in the model's config.json
        - kwargs: keyword arguments that are passed to the corresponding apply_liger_kernel_to_* function.
    """
    if not model_type:
        Logging.info("Model type was not provided. No Liger kernels will be applied.")
        return

    if model_type not in CUSTOM_MODEL_TYPE_TO_APPLY_LIGER_FN.keys():
        Logging.info(
            f"There are currently no Liger kernels supported for model type: {model_type}."
        )
        return

    apply_fn = CUSTOM_MODEL_TYPE_TO_APPLY_LIGER_FN[model_type]
    apply_fn_signature = inspect.signature(apply_fn)

    # Filter out the keyword arguments that are not supported by the apply function
    applicable_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in apply_fn_signature.parameters
    }

    Logging.info(
        f"Applying Liger kernels for model type: {model_type} with kwargs: {applicable_kwargs}"
    )

    # Assume this is invoked pre-model initialization, so we only need to patch transformers code
    apply_fn(**applicable_kwargs)


def _apply_liger_kernel_to_instance(model: PreTrainedModel, **kwargs) -> None:
    """
    Applies Liger kernels to the provided model instance.

    Args:
        - model: the model instance to apply Liger kernels to
        - kwargs: keyword arguments that are passed to the corresponding apply_liger_kernel_to_* function.
    """
    model_type = getattr(model, "config", None) and getattr(
        model.config, "model_type", None
    )

    if not model_type:
        Logging.info(
            "Model type could not be determined from model config. No Liger kernels will be applied."
        )
        return

    if model_type not in CUSTOM_MODEL_TYPE_TO_APPLY_LIGER_FN.keys():
        Logging.info(
            f"There are currently no Liger kernels supported for model type: {model_type}."
        )
        return

    apply_fn = CUSTOM_MODEL_TYPE_TO_APPLY_LIGER_FN[model_type]

    apply_fn_signature = inspect.signature(apply_fn)

    # Filter out the keyword arguments that are not supported by the apply function
    applicable_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in apply_fn_signature.parameters
    }
    Logging.info(
        f"Applying Liger kernels to model instance with model type: {model_type} with kwargs: {applicable_kwargs}"
    )

    apply_fn(model=model, **applicable_kwargs)
