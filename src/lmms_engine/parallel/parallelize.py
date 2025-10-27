from .plan import apply_qwen3_moe_parallel
from .plan.qwen3_moe import unstack_expert_params


class Parallelizer:
    methods = {
        "qwen3_moe": apply_qwen3_moe_parallel,
    }
    revert_methods = {
        "qwen3_moe": unstack_expert_params,
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
    def revert_checkpoint(cls, model, **kwargs):
        # Allow callers to explicitly set the model_type if needed, otherwise
        # fall back to the most recently applied one.
        model_type = kwargs.pop("model_type", None) or cls._model_type
        if model_type is None:
            return
        if model_type not in cls.revert_methods:
            raise ValueError(f"Model type {model_type} not supported")
        return cls.revert_methods[model_type](model, **kwargs)

    @classmethod
    def register_parallelize(cls, model_type, parallelize_fn):
        cls.methods[model_type] = parallelize_fn

    @classmethod
    def register_revert(cls, model_type, revert_fn):
        cls.revert_methods[model_type] = revert_fn
