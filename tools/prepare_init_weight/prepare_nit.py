from lmms_engine.models.nit import NitModel
import torch
from lmms_engine.models.nit.configuration_nit import NitConfig
from lmms_engine.models.nit.modeling_nit import NitModel

def prepare_nit(config: NitConfig, model_safetensors_path, pytorch_dump_folder_path):
    model = NitModel(config)
    state_dict = torch.load(model_safetensors_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Missing keys: {missing}")
    print(f"Unexpected keys: {unexpected}")
    model.save_pretrained(pytorch_dump_folder_path)
    config.save_pretrained(pytorch_dump_folder_path)

if __name__ == "__main__":
    config = NitConfig()
    prepare_nit(config, "/mnt/umm/users/pufanyi/workspace/NiT/checkpoints/nit_xl_model_1000K.safetensors", "pufanyi/NiT")