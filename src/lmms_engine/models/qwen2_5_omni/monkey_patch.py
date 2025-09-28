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
    from liger_kernel.transformers.monkey_patch import (
        _patch_rms_norm_module,
        _patch_swiglu_module,
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
from transformers import PreTrainedModel

from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniThinkerForConditionalGeneration,
)

try:
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniThinkerTextModel
except ImportError:
    Qwen2_5OmniThinkerTextModel = None

try:
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
        Qwen2_5OmniAudioEncoder,
        Qwen2_5OmniVisionEncoder,
    )
except ImportError:
    Qwen2_5OmniAudioEncoder = None
    Qwen2_5OmniVisionEncoder = None

from lmms_engine.parallel.sequence_parallel.ulysses import (
    get_ulysses_sequence_parallel_world_size,
    patch_vlm_for_ulysses_input_slicing,
)

transformer_version = version.parse(transformers.__version__)
SUPPORTED_TRANSFORMER_VERSION = "4.46.1"
TRANSFORMER_DEPRECATION_WARNING = "Support for transformers versions < 4.46.1 will soon be discontinued due to issues with incorrect gradient accumulation. \n Please consider upgrading to avoid potential issues. See details: https://github.com/huggingface/transformers/pull/34191"

from lmms_engine.models.monkey_patch import MONKEY_PATCHER
from lmms_engine.utils.logging_utils import Logging


@MONKEY_PATCHER.register("qwen2_5_omni", "liger")
@MONKEY_PATCHER.register("qwen2_5_omni_thinker", "liger")
def apply_liger_kernel_to_qwen2_5_omni(
    rope: bool = False,  # Disabled by default for Qwen2.5-Omni due to compatibility issues
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None,
    use_rmpad: bool = False,
    freeze_talker: bool = False,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace Qwen2.5-Omni models.
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

    from transformers.models.qwen2_5_omni import modeling_qwen2_5_omni

    from .qwen2_5_omni_liger import lce_forward as qwen2_5_omni_lce_forward

    # Always wrap the forward function with the current configuration
    def wrap_forward(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Add the configuration parameters to kwargs if not already present
            kwargs.setdefault('use_rmpad', use_rmpad)
            kwargs.setdefault('freeze_talker', freeze_talker)
            return func(*args, **kwargs)
        return wrapper

    qwen2_5_omni_lce_forward = wrap_forward(qwen2_5_omni_lce_forward)

    if rope:
        modeling_qwen2_5_omni.apply_multimodal_rotary_pos_emb = (
            liger_multimodal_rotary_pos_emb
        )
    if rms_norm:
        modeling_qwen2_5_omni.Qwen2_5RMSNorm = LigerRMSNorm
    if cross_entropy:
        modeling_qwen2_5_omni.CrossEntropyLoss = LigerCrossEntropyLoss
    if fused_linear_cross_entropy:
        modeling_qwen2_5_omni.Qwen2_5OmniThinkerForConditionalGeneration.forward = qwen2_5_omni_lce_forward
    if swiglu:
        modeling_qwen2_5_omni.Qwen2_5MLP = LigerSwiGLUMLP

    if use_rmpad:
        from .qwen2_5_omni_ops import attn_forward as qwen2_5_omni_attn_forward
        from .qwen2_5_omni_ops import (
            audio_encoder_forward as qwen2_5_omni_audio_encoder_forward,
        )
        from .qwen2_5_omni_ops import (
            audio_encoder_layer_forward as qwen2_5_omni_audio_encoder_layer_forward,
        )
        from .qwen2_5_omni_ops import (
            decoder_layer_forward as qwen2_5_omni_decoder_layer_forward,
        )
        from .qwen2_5_omni_ops import (
            text_model_forward as qwen2_5_omni_text_model_forward,
        )
        from .qwen2_5_omni_ops import (
            vision_encoder_forward as qwen2_5_omni_vision_encoder_forward,
        )
        from .qwen2_5_omni_ops import (
            omni_model_forward as qwen2_5_omni_model_forward,
        )

        # Patch the main model forward
        modeling_qwen2_5_omni.Qwen2_5OmniThinkerModel.forward = qwen2_5_omni_model_forward

        # Use the correct model class names based on investigation
        modeling_qwen2_5_omni.Qwen2_5OmniThinkerTextModel.forward = qwen2_5_omni_text_model_forward
        modeling_qwen2_5_omni.Qwen2_5OmniDecoderLayer.forward = (
            qwen2_5_omni_decoder_layer_forward
        )
        modeling_qwen2_5_omni.Qwen2_5OmniAttention.forward = qwen2_5_omni_attn_forward
        modeling_qwen2_5_omni.Qwen2_5OmniAudioEncoder.forward = qwen2_5_omni_audio_encoder_forward
        modeling_qwen2_5_omni.Qwen2_5OmniAudioEncoderLayer.forward = (
            qwen2_5_omni_audio_encoder_layer_forward
        )
        modeling_qwen2_5_omni.Qwen2_5OmniVisionEncoder.forward = qwen2_5_omni_vision_encoder_forward

    if get_ulysses_sequence_parallel_world_size() > 1:
        patch_vlm_for_ulysses_input_slicing(Qwen2_5OmniThinkerTextModel)

    if model is not None:
        # The model instance already exists, so we need to additionally patch the
        # instance variables that reference already-instantiated modules
        if isinstance(model, Qwen2_5OmniThinkerForConditionalGeneration):
            # The main model structure: model.model is the Qwen2_5OmniThinkerTextModel
            text_model = model.model
            # Audio and vision are handled differently in this architecture
            audio_model = None
            vision_model = None
        elif isinstance(model, Qwen2_5OmniThinkerTextModel):
            text_model = model
            audio_model = None
            vision_model = None
        else:
            # Try to find the text model component
            if hasattr(model, 'model'):
                text_model = model.model
            else:
                text_model = model
            audio_model = None
            vision_model = None

        # Vision and audio encoders are handled differently in Qwen2.5-Omni
        # They may not exist as separate modules to patch

        if text_model is not None:
            if rms_norm and hasattr(text_model, 'norm'):
                _patch_rms_norm_module(text_model.norm)
            if hasattr(text_model, 'layers'):
                for decoder_layer in text_model.layers:
                    if swiglu:
                        _patch_swiglu_module(decoder_layer.mlp, LigerSwiGLUMLP)
                    if rms_norm:
                        _patch_rms_norm_module(decoder_layer.input_layernorm)
                        _patch_rms_norm_module(decoder_layer.post_attention_layernorm)