from .qwen3_moe import apply_qwen3_moe_parallel

MODEL_TO_PARALLEL_METHOD = {
    "qwen3_moe": apply_qwen3_moe_parallel,
}


def apply_parallelize(model, model_type, ep_mesh=None, tp_mesh=None, **kwargs):
    if model_type is None:
        return
    if model_type not in MODEL_TO_PARALLEL_METHOD:
        raise ValueError(f"Model type {model_type} not supported")
    return MODEL_TO_PARALLEL_METHOD[model_type](model, ep_mesh=ep_mesh, tp_mesh=tp_mesh, **kwargs)
