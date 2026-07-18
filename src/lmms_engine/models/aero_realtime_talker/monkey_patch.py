"""Monkey patches for the standalone aero_realtime_talker.

Liger:
  * swap AeroRealtimeTalkerForConditionalGeneration.forward_sub_talker_finetune
    for the LCE version (avoids materializing residual [B, 15, V] logits).

Rmpad:
  * patch the talker trunk to the packed + Ulysses ops:
      - AeroRealtimeTalkerModel.forward             -> talker_model_forward
      - AeroRealtimeTalkerDecoderLayer.forward      -> decoder_layer_forward
      - AeroRealtimeTalkerAttention.forward         -> attn_forward
      - AeroRealtimeTalkerForConditionalGeneration.compute_talker_loss
                                                     -> packed compute_talker_loss
"""

from transformers import PreTrainedModel

from lmms_engine.models.monkey_patch import MONKEY_PATCHER


@MONKEY_PATCHER.register("aero_realtime_talker", "liger")
def apply_liger_kernel_to_aero_realtime_talker(model: PreTrainedModel = None, **kwargs) -> None:
    """Swap the residual-code loss path to fused LCE."""
    from .aero_realtime_talker_liger import lce_forward_sub_talker_finetune
    from .modeling_aero_realtime_talker import (
        AeroRealtimeTalkerForConditionalGeneration,
    )

    if model is None:
        raise ValueError("apply_liger_kernel_to_aero_realtime_talker requires model=...")
    if not isinstance(model, AeroRealtimeTalkerForConditionalGeneration):
        raise TypeError(f"Expected AeroRealtimeTalkerForConditionalGeneration, got {type(model)}")

    AeroRealtimeTalkerForConditionalGeneration.forward_sub_talker_finetune = lce_forward_sub_talker_finetune


@MONKEY_PATCHER.register("aero_realtime_talker", "rmpad")
def apply_rmpad_to_aero_realtime_talker(model: PreTrainedModel = None, **kwargs) -> None:
    """Apply packed and Ulysses forwards to the standalone talker."""
    from lmms_engine.parallel.sequence_parallel.ulysses import (
        get_ulysses_sequence_parallel_world_size,
        patch_vlm_for_ulysses_input_slicing,
    )

    from .aero_realtime_talker_ops import (
        attn_forward,
        compute_talker_loss,
        decoder_layer_forward,
        talker_model_forward,
    )
    from .modeling_aero_realtime_talker import (
        AeroRealtimeTalkerAttention,
        AeroRealtimeTalkerDecoderLayer,
        AeroRealtimeTalkerForConditionalGeneration,
        AeroRealtimeTalkerModel,
    )

    if model is None:
        raise ValueError("apply_rmpad_to_aero_realtime_talker requires model=...")
    if not isinstance(model, AeroRealtimeTalkerForConditionalGeneration):
        raise TypeError(f"Expected AeroRealtimeTalkerForConditionalGeneration, got {type(model)}")

    AeroRealtimeTalkerModel.forward = talker_model_forward
    AeroRealtimeTalkerDecoderLayer.forward = decoder_layer_forward
    AeroRealtimeTalkerAttention.forward = attn_forward
    AeroRealtimeTalkerForConditionalGeneration.compute_talker_loss = compute_talker_loss

    if get_ulysses_sequence_parallel_world_size() > 1:
        patch_vlm_for_ulysses_input_slicing(AeroRealtimeTalkerModel)
