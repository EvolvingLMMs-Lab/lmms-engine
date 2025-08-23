import base64
import glob
import io
import os

from datasets import Dataset, load_dataset, load_from_disk
from huggingface_hub import HfApi, snapshot_download
from PIL import Image

output_path = os.path.join(os.path.dirname(__file__), "data", "blip3o-60k")
dataset = load_from_disk(output_path)

print(dataset)
print(dataset[0])
