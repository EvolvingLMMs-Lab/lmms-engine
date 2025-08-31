import huggingface_hub

model_id = "ByteDance-Seed/BAGEL-7B-MoT"

model_path = huggingface_hub.snapshot_download(
    model_id, local_dir="temp/models/bagel", local_dir_use_symlinks=False
)

print(model_path)

model_path = huggingface_hub.snapshot_download(model_id)

print(model_path)
