from .plan import apply_qwen3_moe_parallel


class Parallelizer:
    methods = {
        "qwen3_moe": apply_qwen3_moe_parallel,
    }

    @staticmethod
    def apply_parallelize(model, model_type, ep_mesh=None, tp_mesh=None, **kwargs):
        if model_type not in Parallelizer.methods:
            raise ValueError(f"Model type {model_type} not supported")
        return Parallelizer.methods[model_type](
            model, ep_mesh=ep_mesh, tp_mesh=tp_mesh, **kwargs
        )

    def register_parallelize(self, model_type, parallelize_fn):
        self.methods[model_type] = parallelize_fn
