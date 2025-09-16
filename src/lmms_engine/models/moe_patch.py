from ..utils.logging_utils import Logging

class MoEParallelPatcher:
    def __init__(self):
        self._dict = {}
        self._dict["model"] = {
            "layers": "layers",
        }
        self._dict["transformer_block"] = {
            "moe_enabled": "moe_enabled",
            "moe": "moe",
            "moe.experts": "moe.experts",
            "moe.router.gate": "moe.router.gate",
            "moe.reorderer": "moe.reorderer",
            "moe.shared_experts.w1": "moe.shared_experts.w1",
            "moe.shared_experts.w2": "moe.shared_experts.w2",
            "moe.shared_experts.w3": "moe.shared_experts.w3",
        }

    def register_model(self, attr_name, moe_stdname):
        if moe_stdname not in self._dict["model"]:
            Logging.warning(
                f"Attribute '{moe_stdname}' not found in model. Available attributes: {self._dict['model'].keys()}"
            )
            return
        self._dict["model"][moe_stdname] = attr_name
        Logging.info(
            f"Registered model attribute '{moe_stdname}' as '{attr_name}' for MoE parallel patching."
        )        
    def register_transformer_block(self, attr_name, moe_stdname):
        if moe_stdname not in self._dict["transformer_block"]:
            Logging.warning(
                f"Attribute '{moe_stdname}' not found in transformer_block. Available attributes: {self._dict['transformer_block'].keys()}"
            )
            return
        self._dict["transformer_block"][moe_stdname] = attr_name
        Logging.info(
            f"Registered transformer_block attribute '{moe_stdname}' as '{attr_name}' for MoE parallel patching."
        )
    def get_attr_name(self, category, moe_stdname):
        if category not in self._dict:
            raise ValueError(f"Invalid category: {category}")
        if moe_stdname not in self._dict[category]:
            raise ValueError(f"Invalid attribute name: {moe_stdname} for category: {category}")
        return self._dict[category][moe_stdname]
    def _apply_ep_tp(self, model, tp_mesh, ep_mesh, ep_tp_mesh, etp_enabled):
        from lmms_engine.parallel.expert_parallel.apply import apply_moe_ep_tp
        apply_moe_ep_tp(
            model=model,
            tp_mesh=tp_mesh,
            ep_mesh=ep_mesh,
            ep_tp_mesh=ep_tp_mesh,
            etp_enabled=etp_enabled,
            model_dict=self._dict["model"],
            transformer_block_dict=self._dict["transformer_block"],
        )
    

MOEPARALLELPATCHER = MoEParallelPatcher()