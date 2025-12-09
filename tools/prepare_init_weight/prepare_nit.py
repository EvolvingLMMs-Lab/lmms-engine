import torch
from safetensors.torch import load_file

from lmms_engine.models.nit import NitModel
from lmms_engine.models.nit.configuration_nit import NitConfig
from lmms_engine.models.nit.modeling_nit import NitModel


def prepare_nit(config: NitConfig, model_safetensors_path, pytorch_dump_folder_path):
    model = NitModel(config)
    if model_safetensors_path.endswith(".safetensors"):
        state_dict = load_file(model_safetensors_path, device="cpu")
    else:
        state_dict = torch.load(model_safetensors_path, map_location="cpu", weights_only=False)
    new_state_dict = {}
    for k, v in state_dict.items():
        if not k.startswith("nit."):
            new_state_dict[f"nit.{k}"] = v
        else:
            new_state_dict[k] = v
    state_dict = new_state_dict
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    print(f"Missing keys: {missing}")
    print(f"Unexpected keys: {unexpected}")
    model.push_to_hub(pytorch_dump_folder_path)
    config.push_to_hub(pytorch_dump_folder_path)


if __name__ == "__main__":
    config = NitConfig()
    prepare_nit(
        config, "/mnt/umm/users/pufanyi/workspace/NiT/checkpoints/nit_xl_model_1000K.safetensors", "pufanyi/NiT"
    )
