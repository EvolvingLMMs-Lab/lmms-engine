import base64
import glob
import io
import os

from datasets import Dataset, load_dataset
from huggingface_hub import snapshot_download
from PIL import Image

def save_image(image: Image.Image, id: str, output_path: str):
    ext = image.format.lower()
    path = os.path.join(output_path, f"images/{id}.{ext}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image.save(path)
    return path
    


if __name__ == "__main__":
    data_path = snapshot_download(repo_id="BLIP3o/BLIP3o-60k", repo_type="dataset")
    data_files = glob.glob(os.path.join(data_path, "*.tar"))
    train_dataset = load_dataset(
        "webdataset", data_files=data_files, split="train", num_proc=64
    )
    # train_dataset.push_to_hub("pufanyi/BLIP3o-60k")

    def generator(dataset, output_path):
        for id, d in enumerate(dataset):
            # print(d)
            img_path = save_image(d["jpg"], id, output_path)
            yield {
                "id": str(id),
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": d["txt"],
                            }
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": img_path,
                                },
                            }
                        ],
                    },
                ],
            }

    # dataset.push_to_hub("pufanyi/BLIP3o-60k", num_proc=64)
    output_path = os.path.join(os.path.dirname(__file__), "data", "blip3o-60k")
    os.makedirs(output_path, exist_ok=True)
    dataset = Dataset.from_generator(generator, gen_kwargs={"dataset": train_dataset, "output_path": output_path}, num_proc=64)
    dataset.save_to_disk(output_path)
