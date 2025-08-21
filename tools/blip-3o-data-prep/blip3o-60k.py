from huggingface_hub import snapshot_download
from datasets import Dataset, load_dataset
import glob
import os
from PIL import Image
import base64
import io

def image_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.convert("RGB").save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_str}"

if __name__ == "__main__":
    data_path = snapshot_download(repo_id='BLIP3o/BLIP3o-60k', repo_type="dataset")
    data_files = glob.glob(os.path.join(data_path, '*.tar'))
    train_dataset = load_dataset("webdataset", data_files=data_files, split="train", num_proc=64)
    # train_dataset.push_to_hub("pufanyi/BLIP3o-60k")
    
    def generator(dataset):
        for id, d in enumerate(dataset):
            # print(d)
            yield {
                "id": str(id),
                "messages": [
                    {
                        "role": "user",
                        "content": [{
                            "type": "text",
                            "text": d["txt"],
                        }],
                    },
                    {
                        "role": "assistant",
                        "content": [{
                            "type": "image_url",
                            "image_url": {
                                "url": image_to_base64(d['jpg']),
                            },
                        }],
                    }
                ]
            }
    
    dataset = Dataset.from_generator(generator, gen_kwargs={"dataset": train_dataset})
    dataset.push_to_hub("pufanyi/BLIP3o-60k", num_proc=64)
