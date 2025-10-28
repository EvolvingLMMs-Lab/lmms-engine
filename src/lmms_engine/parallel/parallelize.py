from .qwen3_moe import apply_qwen3_moe_parallel


class Parallelizer:
    methods = {
        "qwen3_moe": apply_qwen3_moe_parallel,
    }
    _model_type = None

    @classmethod
    def apply_parallelize(cls, model, model_type, ep_mesh=None, tp_mesh=None, **kwargs):
        if model_type is None:
            return
        if model_type not in cls.methods:
            raise ValueError(f"Model type {model_type} not supported")
        cls._model_type = model_type
        return cls.methods[model_type](model, ep_mesh=ep_mesh, tp_mesh=tp_mesh, **kwargs)

    @classmethod
    def register_parallelize(cls, model_type, parallelize_fn):
        cls.methods[model_type] = parallelize_fn
