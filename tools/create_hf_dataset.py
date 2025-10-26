import argparse
import json
import os
from multiprocessing import cpu_count

from datasets import load_dataset
from tqdm import tqdm


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="lmms-lab/LLaVA-NeXT-Data")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--repo-id", type=str)

    return parser.parse_args()


def transforms(examples):
    chunks = convert_llava_to_openai(examples["conversations"])

    return {"messages": chunks}


def convert_llava_to_openai(content):
    messages = []
    for item in content:
        if item["from"] == "human":
            content = []
            if "<image>" in item["value"]:
                content.append({"type": "image_url", "image_url": {"url": "<image>"}})
            content.append({"type": "text", "text": item["value"].replace("<image>", "")})
            messages.append({"role": "user", "content": content})
        elif item["from"] == "gpt":
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": item["value"]}],
                }
            )

    return messages


if __name__ == "__main__":
    args = parse_argument()
    dataset = load_dataset(args.dataset_path, split=args.split)
    dataset = dataset.map(transforms, num_proc=cpu_count() // 2)
    dataset = dataset.remove_columns(["conversations"])
    dataset.push_to_hub(args.repo_id)
