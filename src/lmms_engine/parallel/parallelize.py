from .plan import apply_qwen3_moe_parallel, unstack_expert_params

class Parallelizer:
    methods = {
        "qwen3_moe": apply_qwen3_moe_parallel,
    }
    revert_methods = {
        "qwen3_moe": unstack_expert_params,
    }

    def apply_parallelize(self, model, model_type, ep_mesh=None, tp_mesh=None, **kwargs):
        if model_type is None:
            return
        if model_type not in Parallelizer.methods:
            raise ValueError(f"Model type {model_type} not supported")
        self.model_type = model_type
        return Parallelizer.methods[model_type](
            model, ep_mesh=ep_mesh, tp_mesh=tp_mesh, **kwargs
        )
    
    def revert_checkpoint(self, model, **kwargs):
        model_type = self.model_type
        if model_type is None:
            return
        if model_type not in Parallelizer.revert_methods:
            raise ValueError(f"Model type {model_type} not supported")
        return Parallelizer.revert_methods[model_type](model, **kwargs)

    def register_parallelize(self, model_type, parallelize_fn):
        self.methods[model_type] = parallelize_fn
    def register_revert(self, model_type, revert_fn):
        self.revert_methods[model_type] = revert_fn
