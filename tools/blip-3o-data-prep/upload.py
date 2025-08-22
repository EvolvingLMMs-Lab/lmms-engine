import base64
import glob
import io
import os

from datasets import Dataset, load_dataset
from huggingface_hub import HfApi, snapshot_download
from PIL import Image

output_path = os.path.join(os.path.dirname(__file__), "data", "blip3o-60k")
api = HfApi()
repo_id = "pufanyi/BLIP3o-60k"
api.upload_large_folder(
    folder_path=output_path,
    repo_id=repo_id,
    repo_type="dataset",
)
print(f"Dataset successfully uploaded to https://huggingface.co/datasets/{repo_id}")
