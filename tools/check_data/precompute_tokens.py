import argparse
import json
import math
import os
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

import jsonlines
import librosa
import yaml
from decord import VideoReader, cpu
from PIL import Image, PngImagePlugin
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

AUDIO_TOKENS_PER_SECOND = 25  # 750 / 30
VIDEO_TOKENS_PER_FRAMES = (360 * 420) / (
    14 * 14 * 4
)  # 360x420 image, 14x14 per patch, 4 patches per token
LARGE_ENOUGH_NUMBER = 1000
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml-file", type=str, help="path to your yaml file")

    return parser.parse_args()


def count_audio_tokens(audio_path):
    try:
        y, sr = librosa.load(audio_path)
        total_duration_in_seconds = librosa.get_duration(y=y, sr=sr)
        num_tokens = math.ceil(total_duration_in_seconds * AUDIO_TOKENS_PER_SECOND)
    except Exception as e:
        print(f"Error: {str(e)} when loading {audio_path}")
        num_tokens = 750  # Default to 30 seconds
    return num_tokens


def calculate_image_tokens(image_path):
    try:
        image = Image.open(image_path)
        width, height = image.size
        num_tokens = (
            width * height // (14 * 14 * 4)
        )  # Use Qwen Navit estimation (14, 14) per patch, 4 patches per token
    except Exception as e:
        print(f"Error: {str(e)} when loading {image_path}")
        num_tokens = 500 * 500 // (14 * 14 * 4)  # Default to 500x500 image
    return num_tokens


def calculate_video_tokens(video_path):
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames, video_fps = len(vr), vr.get_avg_fps()
        num_tokens = total_frames / video_fps * VIDEO_TOKENS_PER_FRAMES
    except Exception as e:
        print(f"Error: {str(e)} when loading {video_path}")
        num_tokens = 32 * VIDEO_TOKENS_PER_FRAMES  # Default to 32 frames
    return num_tokens


def check_data_exists(data_dict):
    data = data_dict["data"]
    data_folder = data_dict["data_folder"]

    messages = data["messages"]
    for message in messages:
        for content in message["content"]:
            if content["type"] == "image_url":
                tokens = calculate_image_tokens(
                    os.path.join(data_folder, content["image_url"]["url"])
                )
                content["precomputed_tokens"] = tokens
            elif content["type"] == "audio_url":
                tokens = count_audio_tokens(
                    os.path.join(data_folder, content["audio_url"]["url"])
                )
                content["precomputed_tokens"] = tokens
            elif content["type"] == "video_url":
                tokens = calculate_video_tokens(
                    os.path.join(data_folder, content["video_url"]["url"])
                )
                content["precomputed_tokens"] = tokens

    if "id" not in data:
        data["id"] = "Unknown_ID"
    return {"id": data["id"], "messages": messages}


def check_single_dataset(info):
    data_path, data_folder, data_type = info
    if data_type == "json":
        with open(data_path, "r") as f:
            data = json.load(f)
    elif data_type == "jsonl":
        data = []
        with jsonlines.open(data_path, "r") as f:
            for d in f:
                data.append(d)

    data_folder = [data_folder] * len(data)
    data_dict = [
        {"data_folder": data_folder, "data": d}
        for data_folder, d in zip(data_folder, data)
    ]

    with ThreadPool(32) as p:
        results = list(
            tqdm(
                p.imap(check_data_exists, data_dict),
                total=len(data_dict),
                desc=f"Dataset {os.path.basename(data_path)}",
            )
        )

    with jsonlines.open(data_path, "w") as writer:
        writer.write_all(results)


if __name__ == "__main__":
    args = parse_argument()
    data_list = []
    data_folder_list = []
    with open(args.yaml_file, "r") as f:
        yaml_data = yaml.safe_load(f)
        datasets = yaml_data.get("datasets")
        data_paths = [dataset.get("path") for dataset in datasets]
        data_folders = [dataset.get("data_folder") for dataset in datasets]
        data_types = [dataset.get("data_type") for dataset in datasets]

    info = [
        (data_path, data_folder, data_type)
        for data_path, data_folder, data_type in zip(
            data_paths, data_folders, data_types
        )
    ]
    with Pool(32) as p:
        results = list(tqdm(p.imap(check_single_dataset, info), total=len(info)))
