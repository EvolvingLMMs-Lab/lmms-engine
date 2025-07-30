import argparse
import json
import math
import os
from io import BytesIO
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
from typing import Literal

import jsonlines
import librosa
import soundfile as sf
import yaml
from decord import VideoReader, cpu
from PIL import Image, PngImagePlugin
from tqdm import tqdm

AUDIO_TOKENS_PER_SECOND = 25  # 750 / 30
VIDEO_TOKENS_PER_FRAMES = (360 * 420) / (
    14 * 14 * 4
)  # 360x420 image, 14x14 per patch, 4 patches per token
LARGE_ENOUGH_NUMBER = 1000
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)


BUCKET_NAME = os.environ.get("BUCKET_NAME", "YOUR_BUCKET_NAME")
CLOUD_SOURCE = os.environ.get("CLOUD_SOURCE", "gcs")

if CLOUD_SOURCE == "gcs":
    try:
        from google.cloud.storage import Client

        storage_client = Client()
    except ImportError:
        print("Google Cloud SDK not installed. Skipping import.")
elif CLOUD_SOURCE == "azure":
    try:
        from azure.identity import DefaultAzureCredential
        from azure.storage.blob import (
            BlobPrefix,
            BlobServiceClient,
            ContainerClient,
            LinearRetry,
        )

        RETRY_POLICY = LinearRetry(backoff=10, retry_total=5, random_jitter_range=0)
        SAS_URL = os.environ.get("AZURE_STORAGE_SAS_URL", "YOUR_SAS_URL")
        storage_client = BlobServiceClient(
            account_url=SAS_URL, retry_policy=RETRY_POLICY
        )
    except ImportError:
        print("Azure SDK not installed. Skipping import.")


def download_blob_to_stream(
    storage_client,
    bucket_name: str,
    source_blob_name: str,
    file_obj: BytesIO,
    storage_type: Literal["gcs", "azure"] = "azure",
    max_retries: int = 5,
) -> BytesIO:
    for i in range(max_retries):
        try:
            if storage_type == "gcs":
                bucket = storage_client.bucket(bucket_name)
                blob = bucket.blob(source_blob_name)
                blob.download_to_file(file_obj)
            elif storage_type == "azure":
                blob_client = storage_client.get_blob_client(
                    container=bucket_name, blob=source_blob_name
                )
                blob_client.download_blob().readinto(file_obj)
            break
        except Exception as e:
            print(f"Attempt {i} Error downloading blob: {source_blob_name}")
            print(f"Error: {e}")
            print(f"Retrying ...")

    return file_obj


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml-file", type=str, help="path to your yaml file")

    return parser.parse_args()


def get_bytes_from_file(file_path):
    file_obj = BytesIO()
    file_obj = download_blob_to_stream(
        storage_client,
        BUCKET_NAME,
        file_path,
        file_obj,
        CLOUD_SOURCE,
    )
    file_obj.seek(0)
    return file_obj


def count_audio_tokens(audio_path):
    file_obj = get_bytes_from_file(audio_path)
    try:
        audio, orig_sr = sf.read(file_obj)
        # This is an 2d array, so we need to convert it to 1d
        # Convert the left and right channel to 1
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        total_duration_in_seconds = librosa.get_duration(y=audio, sr=orig_sr)
        num_tokens = math.ceil(total_duration_in_seconds * AUDIO_TOKENS_PER_SECOND)
    except Exception as e:
        print(f"Error: {str(e)} when loading {audio_path}")
        num_tokens = 750  # Default to 30 seconds
    return num_tokens


def calculate_image_tokens(image_path):
    file_obj = get_bytes_from_file(image_path)
    try:
        image = Image.open(file_obj)
        width, height = image.size
        num_tokens = (
            width * height // (14 * 14 * 4)
        )  # Use Qwen Navit estimation (14, 14) per patch, 4 patches per token
    except Exception as e:
        print(f"Error: {str(e)} when loading {image_path}")
        num_tokens = 500 * 500 // (14 * 14 * 4)  # Default to 500x500 image
    return num_tokens


def calculate_video_tokens(video_path):
    file_obj = get_bytes_from_file(video_path)
    try:
        vr = VideoReader(file_obj, ctx=cpu(0))
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
