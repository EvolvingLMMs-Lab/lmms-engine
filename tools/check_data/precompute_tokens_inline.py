import argparse
import math
import os
from functools import partial

import librosa
from decord import VideoReader, cpu
from PIL import Image, PngImagePlugin

from lmms_engine.utils import DataUtilities, config_loader

AUDIO_TOKENS_PER_SECOND = 25  # 750 / 30
VIDEO_TOKENS_PER_FRAMES = (360 * 420) / (
    14 * 14 * 4
)  # 360x420 image, 14x14 per patch, 4 patches per token
LARGE_ENOUGH_NUMBER = 1000
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml-file", type=str, help="path to your yaml file")
    parser.add_argument("--output-file", type=str, help="path to your output file")

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


def compute_tokens(example, indice, dataset_folder):
    messages = example["messages"]
    data_folder = dataset_folder[indice]
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
            elif content["type"] == "text":
                content["precomputed_tokens"] = len(content["text"].split(" ")) * 1.25

    if "id" not in example:
        example["id"] = "Unknown_ID"
    return {"id": example["id"], "messages": messages}


if __name__ == "__main__":
    args = parse_argument()
    dataset = config_loader.load_config(args.yaml_file)
    dataset_yaml = dataset[0]["config"]["dataset_config"]["datasets"]

    dataset_list, dataset_folder = DataUtilities.load_inline_datasets(dataset_yaml)
    fn = partial(compute_tokens, dataset_folder=dataset_folder)
    dataset_list = dataset_list.map(
        fn, with_indices=True, num_proc=8, remove_columns=["messages"]
    )
    dataset_list.to_parquet(args.output_file)
