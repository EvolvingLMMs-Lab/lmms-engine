from lmms_engine.train.config import TrainingArguments

from .qwen3_moe.parallelize import apply_qwen3_moe_parallelize_fn

# Lazy import to avoid circular dependency
def _get_qwen3_omni_moe_parallelize_fn():
    from .qwen3_omni_moe.parallelize import apply_qwen3_omni_moe_parallelize_fn
    return apply_qwen3_omni_moe_parallelize_fn

MODEL_TO_PARALLEL_METHOD = {
    "qwen3_moe": apply_qwen3_moe_parallelize_fn,
    "qwen3_omni_moe": lambda *args, **kwargs: _get_qwen3_omni_moe_parallelize_fn()(*args, **kwargs),
    "qwen3_omni_moe_thinker": lambda *args, **kwargs: _get_qwen3_omni_moe_parallelize_fn()(*args, **kwargs),
}


def apply_parallelize(model, model_type, train_args: TrainingArguments, **kwargs):
    """
    Apply parallelization based on model type.

    Args:
        model: The model to parallelize
        model_type: Key in MODEL_TO_PARALLEL_METHOD (e.g., "qwen3_moe")
        train_args: Training configuration

    Raises:
        ValueError: If model_type is not supported
    """
    if model_type not in MODEL_TO_PARALLEL_METHOD:
        raise ValueError(f"Model type {model_type} not supported")

    return MODEL_TO_PARALLEL_METHOD[model_type](model, train_args, **kwargs)
