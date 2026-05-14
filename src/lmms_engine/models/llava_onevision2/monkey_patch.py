"""Monkey patches for LlavaOnevision2 (auto_map / trust_remote_code model).

Because OV2 modeling classes are loaded dynamically from the checkpoint,
we cannot patch their classes module-globally. All patches here are applied
at the **instance** level on a materialized model.

Two independent patch_types are registered:

  * ``"liger"``  – Liger kernels: RoPE, RMSNorm, SwiGLU MLP, vision LayerNorm,
                    and the fused linear cross-entropy loss bound onto the
                    OV2 ``ForConditionalGeneration.forward``.
  * ``"rmpad"``  – Sequence-packing (unpadded) attention path: class-level
                    patches to the inner Qwen3 layers (so attention/decoder/
                    model forwards consume ``cu_seq_lens`` + ``indices``) and
                    a CE-loss-flavoured ``causal_lm_forward`` that shifts
                    per-seq using ``seq_lens``.

The runner applies these in order: ``["liger", "rmpad"]`` when both are
requested, so ``rmpad``'s rebinding runs *after* ``liger``'s and detects the
already-installed ``loss_fn="lce"`` to preserve it.

Stacked behaviour:

  * ``liger`` alone  → fused LCE, no unpadding.
  * ``rmpad`` alone  → unpacked attention, standard CE loss.
  * ``liger`` + ``rmpad`` (runner order, rmpad rebinds last) → unpacked +
    fused LCE (the historical default of this codebase).
  * Neither         → stock HF forward.
"""

from functools import partial
from types import MethodType

from loguru import logger
from transformers import PreTrainedModel

try:
    from liger_kernel.transformers.monkey_patch import (
        _patch_layer_norm_module,
        _patch_rms_norm_module,
        _patch_swiglu_module,
    )
    from liger_kernel.transformers.swiglu import LigerSwiGLUMLP
except ImportError:
    _patch_layer_norm_module = None
    _patch_rms_norm_module = None
    _patch_swiglu_module = None
    LigerSwiGLUMLP = None
    logger.warning("liger kernel not installed; OV2 liger patch will be a no-op.")

from lmms_engine.models.monkey_patch import MONKEY_PATCHER

from .llava_onevision2_ops import _register_ov2_module, causal_lm_forward, model_forward

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bind_causal_lm_forward(model, *, loss_fn: str, use_rmpad: bool) -> None:
    """Bind ``causal_lm_forward`` onto the OV2 CausalLM instance with patch-time
    options fixed via :func:`functools.partial`.

    If called multiple times (e.g. once by ``liger`` then by ``rmpad``), the
    last call wins — the matrix in the module docstring relies on this.
    """
    bound = partial(causal_lm_forward, loss_fn=loss_fn, use_rmpad=use_rmpad)

    # ``MethodType`` requires a function; wrap the partial to expose ``self``.
    def _forward(self, *args, **kwargs):
        return bound(self, *args, **kwargs)

    model.forward = MethodType(_forward, model)


def _bind_outer_model_forward(model) -> None:
    """Bind OV2 ``LlavaOnevision2Model.forward`` (vision injection + LM call)."""
    ov2_model = getattr(model, "model", None)
    if ov2_model is None:
        logger.warning("OV2: model.model not found; cannot bind outer model_forward.")
        return
    _register_ov2_module(model)
    ov2_model.forward = MethodType(model_forward, ov2_model)


def _patch_qwen3_text_submodules(language_model, *, rms_norm: bool, swiglu: bool) -> None:
    """Instance-level swaps of RMSNorm / SwiGLU modules in an already-loaded
    Qwen3Model. The class-level Liger patches only affect *future* construction;
    we must mutate existing instances explicitly."""
    if language_model is None:
        return
    if _patch_rms_norm_module is None:
        return

    if rms_norm and hasattr(language_model, "norm"):
        _patch_rms_norm_module(language_model.norm)

    for decoder_layer in getattr(language_model, "layers", []):
        if rms_norm:
            if hasattr(decoder_layer, "input_layernorm"):
                _patch_rms_norm_module(decoder_layer.input_layernorm)
            if hasattr(decoder_layer, "post_attention_layernorm"):
                _patch_rms_norm_module(decoder_layer.post_attention_layernorm)
            self_attn = getattr(decoder_layer, "self_attn", None)
            if self_attn is not None:
                if getattr(self_attn, "q_norm", None) is not None:
                    _patch_rms_norm_module(self_attn.q_norm)
                if getattr(self_attn, "k_norm", None) is not None:
                    _patch_rms_norm_module(self_attn.k_norm)
        if swiglu and _patch_swiglu_module is not None and LigerSwiGLUMLP is not None:
            if hasattr(decoder_layer, "mlp"):
                _patch_swiglu_module(decoder_layer.mlp, LigerSwiGLUMLP)


def _patch_ov2_vision_layer_norms(visual) -> None:
    """Replace LayerNorm modules inside the OV2 vision encoder with Liger's."""
    if visual is None or _patch_layer_norm_module is None:
        return
    if hasattr(visual, "layernorm_pre"):
        _patch_layer_norm_module(visual.layernorm_pre)
    if getattr(visual, "layernorm_post", None) is not None:
        _patch_layer_norm_module(visual.layernorm_post)
    encoder = getattr(visual, "encoder", None)
    if encoder is None:
        return
    for vlayer in getattr(encoder, "layers", []):
        if hasattr(vlayer, "layer_norm1"):
            _patch_layer_norm_module(vlayer.layer_norm1)
        if hasattr(vlayer, "layer_norm2"):
            _patch_layer_norm_module(vlayer.layer_norm2)


# ---------------------------------------------------------------------------
# Public entry points (registered with MONKEY_PATCHER)
# ---------------------------------------------------------------------------


@MONKEY_PATCHER.register("llava_onevision2", "liger")
def apply_liger_kernel_to_llava_onevision2(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    layer_norm: bool = True,
    model: PreTrainedModel = None,
) -> None:
    """Apply *only* the Liger-kernel patches to an OV2 model instance.

    Does not touch the attention / decoder forwards (those are owned by the
    ``"rmpad"`` patch). Binds ``causal_lm_forward(loss_fn="lce", use_rmpad=False)``
    so the LM head loss runs as fused linear CE without materializing logits.

    ``model`` is required: OV2 is auto_map / trust_remote_code, so we have no
    shared class to mutate.
    """
    if model is None:
        logger.warning("OV2 liger patch skipped: no model instance passed.")
        return

    # ----- 1. Class-level Liger patches for the inner Qwen3 LM ----------------
    # Reuse the qwen3 patch but force its rmpad / fused-LCE bits OFF so we
    # don't accidentally rebind ``Qwen3ForCausalLM.forward`` or set up unpad
    # ops we don't want yet.
    from lmms_engine.models.qwen3.monkey_patch import apply_liger_kernel_to_qwen3

    apply_liger_kernel_to_qwen3(
        rope=rope,
        cross_entropy=cross_entropy,
        fused_linear_cross_entropy=False,  # we bind OV2's CausalLM forward below
        rms_norm=rms_norm,
        swiglu=swiglu,
        model=None,
        use_rmpad=False,
    )

    # ----- 2. Instance-level Liger swaps in already-loaded submodules ---------
    ov2_model = getattr(model, "model", None)
    if ov2_model is None:
        logger.warning("OV2 liger patch: model.model not found; aborting.")
        return
    _patch_qwen3_text_submodules(
        getattr(ov2_model, "language_model", None),
        rms_norm=rms_norm,
        swiglu=swiglu,
    )
    if layer_norm:
        _patch_ov2_vision_layer_norms(getattr(ov2_model, "visual", None))

    # ----- 3. Bind OV2 causal LM forward with fused LCE -----------------------
    # The ``"rmpad"`` patch (if requested) is applied separately *after* this
    # one by the runner; it will rebind ``causal_lm_forward`` with
    # ``use_rmpad=True`` while detecting and preserving ``loss_fn="lce"``.
    if fused_linear_cross_entropy:
        _register_ov2_module(model)  # cache OV2 output classes pre-FSDP
        _bind_causal_lm_forward(model, loss_fn="lce", use_rmpad=False)


@MONKEY_PATCHER.register("llava_onevision2", "rmpad")
def apply_rmpad_to_llava_onevision2(
    model: PreTrainedModel = None,
) -> None:
    """Apply *only* the rmpad (sequence-packing) patches to an OV2 model.

    Patches the inner Qwen3 attention/decoder/model forwards class-level to
    consume ``cu_seq_lens`` + ``indices``, binds OV2 ``model_forward`` (which
    propagates rmpad metadata out to ``causal_lm_forward``), and binds
    ``causal_lm_forward(loss_fn="ce", use_rmpad=True)``.

    When stacked on top of the ``"liger"`` patch, this overrides the latter's
    ``causal_lm_forward`` binding so the loss becomes ``loss_fn="lce" +
    use_rmpad=True`` (handled by the caller passing both patches; the stacking
    behaviour is documented in the matrix in the module docstring).
    """
    if model is None:
        logger.warning("OV2 rmpad patch skipped: no model instance passed.")
        return

    # ----- 1. Class-level rmpad patches for inner Qwen3 layers ----------------
    # We piggy-back on qwen3's apply function with everything else disabled.
    from lmms_engine.models.qwen3.monkey_patch import apply_liger_kernel_to_qwen3

    apply_liger_kernel_to_qwen3(
        rope=False,
        cross_entropy=False,
        fused_linear_cross_entropy=False,
        rms_norm=False,
        swiglu=False,
        model=None,
        use_rmpad=True,  # this is the bit we actually want
    )

    # ----- 2. Outer OV2 model_forward + causal_lm_forward bindings ------------
    _bind_outer_model_forward(model)

    # If liger already bound a fused-LCE forward, preserve that and just flip
    # ``use_rmpad=True``. Otherwise bind a plain-CE rmpad forward.
    current_loss_fn = _detect_bound_loss_fn(model)
    _bind_causal_lm_forward(
        model,
        loss_fn=current_loss_fn or "ce",
        use_rmpad=True,
    )


def _detect_bound_loss_fn(model) -> str:
    """Inspect ``model.forward`` to recover ``loss_fn`` if we previously bound
    it via :func:`_bind_causal_lm_forward`. Returns ``None`` if no prior bind
    can be detected (e.g. stock HF forward)."""
    fwd = getattr(model, "forward", None)
    if fwd is None:
        return None
    # ``MethodType(_forward, model)`` exposes the underlying function on .__func__.
    inner = getattr(fwd, "__func__", None)
    if inner is None:
        return None
    # The closure of ``_forward`` captures ``bound = partial(...)``; pull it out.
    closure = getattr(inner, "__closure__", None) or ()
    for cell in closure:
        try:
            val = cell.cell_contents
        except ValueError:
            continue
        if isinstance(val, partial):
            return val.keywords.get("loss_fn")
    return None
