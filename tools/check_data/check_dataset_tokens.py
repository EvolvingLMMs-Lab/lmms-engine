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
from PIL import Image
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

AUDIO_TOKENS_PER_SECOND = 25  # 750 / 30
VIDEO_TOKENS_PER_FRAMES = (360 * 420) / (
    14 * 14 * 4
)  # 360x420 image, 14x14 per patch, 4 patches per token


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

    images_list = []
    audios_list = []
    videos_list = []
    messages = data["messages"]
    for message in messages:
        for content in message["content"]:
            if content["type"] == "image_url":
                images_list.append(content["image_url"]["url"])
            elif content["type"] == "audio_url":
                audios_list.append(content["audio_url"]["url"])
            elif content["type"] == "video_url":
                videos_list.append(content["video_url"]["url"])

    num_tokens = 0
    for path in images_list:
        num_tokens += calculate_image_tokens(os.path.join(data_folder, path))
    for path in audios_list:
        num_tokens += count_audio_tokens(os.path.join(data_folder, path))
    for path in videos_list:
        num_tokens += calculate_video_tokens(os.path.join(data_folder, path))

    return num_tokens


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

    tokens_in_millions = sum(results) / 1e6

    print(
        f"\n\nDataset {data_path}, \n Estimated Total tokens: {tokens_in_millions:.2f}M\n\n"
    )

    return results


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

    # Save into a file
    with open("tokens.txt", "w") as f:
        for data_path, result in zip(data_paths, results):
            f.write(f"Dataset {data_path}, \nEstimated Total tokens: {sum(result)}\n\n")
