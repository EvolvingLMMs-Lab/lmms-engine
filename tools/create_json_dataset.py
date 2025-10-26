import argparse
import json
import os

import jsonlines
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, default="lmms-lab/LLaVA-NeXT-Data")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output-folder", type=str, default="./data")
    parser.add_argument("--output-name", type=str, default="lmms_engine.json", help="output file name")
    parser.add_argument("--modalities", type=str, default="image", choices=["image", "audio"])
    parser.add_argument("--index-column", type=str, default="id", help="The index column of the dataset")
    parser.add_argument(
        "--media-column",
        type=str,
        default="image",
        help="The media column of the dataset",
    )
    parser.add_argument(
        "--conv-column",
        type=str,
        default="conversations",
        help="The conversation column of the dataset",
    )
    parser.add_argument("--type", type=str, default="json", choices=["json", "jsonl"])

    return parser.parse_args()


def convert_llava_to_openai(content, image_path: str):
    messages = []
    for item in content:
        if item["from"] == "human":
            content = []
            if "<image>" in item["value"]:
                content.append({"type": "image_url", "image_url": {"url": image_path}})
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


def construct_audio_messages(question, audio_path):
    messages = []
    content = []
    content.append({"type": "audio_url", "audio_url": {"url": audio_path}})
    messages.append({"role": "user", "content": content})
    messages.append(
        {
            "role": "assistant",
            "content": [{"type": "text", "text": question}],
        }
    )
    return messages


def save_image(image, image_path):
    image.convert("RGB").save(image_path)


def save_audio(audio_path, array, sampling_rate):
    sf.write(audio_path, array, sampling_rate)


def handle_single_audio(data_dict):
    ids = data_dict["id"]
    item = dataset[ids]
    index_column = data_dict["index_column"]
    media_column = data_dict["media_column"]
    conv_column = data_dict["conv_column"]
    audio_path = os.path.join(args.output_folder, "audio", item[index_column] + ".wav")
    save_audio(audio_path, item[media_column]["array"], item[media_column]["sampling_rate"])
    messages = construct_audio_messages(item[conv_column].replace("Omni", "Kino"), audio_path)
    return {"id": item[index_column], "messages": messages}


if __name__ == "__main__":
    args = parse_argument()
    dataset = load_dataset(args.dataset_path, split=args.split)
    index_column = args.index_column
    media_column = args.media_column
    conv_column = args.conv_column
    os.makedirs(args.output_folder, exist_ok=True)
    os.makedirs(os.path.join(args.output_folder, args.modalities), exist_ok=True)
    json_dataset = []
    pbar = tqdm(total=len(dataset), desc="Saving...")
    if args.modalities == "image":
        for item in dataset:
            image_path = os.path.join(args.output_folder, "image", item[index_column] + ".jpg")
            if item[media_column] is not None:
                item[media_column].convert("RGB").save(image_path)
            messages = convert_llava_to_openai(item[conv_column], image_path)
            json_dataset.append({"id": item["id"], "messages": messages})
            pbar.update(1)
    elif args.modalities == "audio":
        data_dict_list = [
            {
                "id": idx,
                "index_column": index_column,
                "media_column": media_column,
                "conv_column": conv_column,
            }
            for idx in range(len(dataset))
        ]
        json_dataset = process_map(handle_single_audio, data_dict_list, max_workers=32)
        pass

    if args.type == "json":
        with open(os.path.join(args.output_folder, args.output_name), "w") as f:
            json.dump(json_dataset, f, indent=4)
    elif args.type == "jsonl":
        if args.output_name.endswith("json"):
            output_name = args.output_name.replace("json", "jsonl")
        with jsonlines.open(os.path.join(args.output_folder, output_name), "w") as f:
            f.write_all(json_dataset)
