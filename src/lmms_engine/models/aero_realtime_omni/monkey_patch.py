"""Monkey patches for the aero_realtime_omni wrapper.

Liger:
  * apply_liger_kernel_to_aero_realtime on the wrapper's thinker
  * swap AeroRealtimeTalkerForConditionalGeneration.forward_sub_talker_finetune
    for the LCE version (avoids materializing residual [B, 15, V] logits).

Rmpad:
  * apply_rmpad_to_aero_realtime on the thinker (as before)
  * patch the talker trunk to the packed + Ulysses ops:
      - AeroRealtimeTalkerModel.forward             -> talker_model_forward
      - AeroRealtimeTalkerDecoderLayer.forward      -> decoder_layer_forward
      - AeroRealtimeTalkerAttention.forward         -> attn_forward
      - AeroRealtimeOmniForConditionalGeneration.compute_talker_loss
                                                     -> packed compute_talker_loss
"""

from transformers import PreTrainedModel

from lmms_engine.models.aero_realtime.monkey_patch import (
    apply_liger_kernel_to_aero_realtime,
    apply_rmpad_to_aero_realtime,
)
from lmms_engine.models.monkey_patch import MONKEY_PATCHER


@MONKEY_PATCHER.register("aero_realtime_omni", "liger")
def apply_liger_kernel_to_aero_realtime_omni(model: PreTrainedModel = None, **kwargs) -> None:
    """Apply liger to the thinker and swap talker's residual-loss path to LCE."""
    from .aero_realtime_omni_liger import lce_forward_sub_talker_finetune
    from .modeling_aero_realtime_omni import AeroRealtimeOmniForConditionalGeneration
    from .modeling_aero_realtime_talker import (
        AeroRealtimeTalkerForConditionalGeneration,
    )

    if model is None:
        raise ValueError("apply_liger_kernel_to_aero_realtime_omni requires model=...")
    if not isinstance(model, AeroRealtimeOmniForConditionalGeneration):
        raise TypeError(f"Expected AeroRealtimeOmniForConditionalGeneration, got {type(model)}")
    apply_liger_kernel_to_aero_realtime(model=model.thinker, **kwargs)

    AeroRealtimeTalkerForConditionalGeneration.forward_sub_talker_finetune = lce_forward_sub_talker_finetune


@MONKEY_PATCHER.register("aero_realtime_omni", "rmpad")
def apply_rmpad_to_aero_realtime_omni(model: PreTrainedModel = None, **kwargs) -> None:
    """Rmpad the thinker (as before) + packed+Ulysses forward on the talker."""
    from lmms_engine.parallel.sequence_parallel.ulysses import (
        get_ulysses_sequence_parallel_world_size,
        patch_vlm_for_ulysses_input_slicing,
    )

    from .aero_realtime_omni_ops import (
        attn_forward,
        compute_talker_loss,
        decoder_layer_forward,
        talker_model_forward,
    )
    from .modeling_aero_realtime_omni import AeroRealtimeOmniForConditionalGeneration
    from .modeling_aero_realtime_talker import (
        AeroRealtimeTalkerAttention,
        AeroRealtimeTalkerDecoderLayer,
        AeroRealtimeTalkerModel,
    )

    if model is None:
        raise ValueError("apply_rmpad_to_aero_realtime_omni requires model=...")
    if not isinstance(model, AeroRealtimeOmniForConditionalGeneration):
        raise TypeError(f"Expected AeroRealtimeOmniForConditionalGeneration, got {type(model)}")
    apply_rmpad_to_aero_realtime(model=model.thinker, **kwargs)

    AeroRealtimeTalkerModel.forward = talker_model_forward
    AeroRealtimeTalkerDecoderLayer.forward = decoder_layer_forward
    AeroRealtimeTalkerAttention.forward = attn_forward
    AeroRealtimeOmniForConditionalGeneration.compute_talker_loss = compute_talker_loss

    if get_ulysses_sequence_parallel_world_size() > 1:
        patch_vlm_for_ulysses_input_slicing(AeroRealtimeTalkerModel)
