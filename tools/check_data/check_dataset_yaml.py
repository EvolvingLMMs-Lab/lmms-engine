import argparse
import json
import os
from multiprocessing import Pool

import jsonlines
import yaml
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml-file", type=str, help="path to your yaml file")

    return parser.parse_args()


def check_data_exists(data_dict):
    data = data_dict["data"]
    data_folder = data_dict["data_folder"]

    images_list = []
    audios_list = []
    messages = data["messages"]
    for message in messages:
        for content in message["content"]:
            if content["type"] == "image_url":
                images_list.append(content["image_url"]["url"])
            elif content["type"] == "audio_url":
                audios_list.append(content["audio_url"]["url"])

    not_exists = []
    for path in images_list:
        if not os.path.exists(os.path.join(data_folder, path)):
            not_exists.append(path)
    for path in audios_list:
        if not os.path.exists(os.path.join(data_folder, path)):
            not_exists.append(path)
            print(os.path.join(data_folder, path))
    return not_exists


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

    for data_path, data_folder, data_type in zip(data_paths, data_folders, data_types):
        if data_type == "json":
            with open(data_path, "r") as f:
                data = json.load(f)
                data_list.extend(data)
                data_folder_list.extend([data_folder] * len(data))
        elif data_type == "jsonl":
            cur_data_dict = []
            with open(data_path, "r") as json_file:
                for line in json_file:
                    cur_data_dict.append(json.loads(line.strip()))
            data_list.extend(cur_data_dict)
            data_folder_list.extend([data_folder] * len(cur_data_dict))

    data_dict = [
        {"data_folder": data_folder, "data": data}
        for data_folder, data in zip(data_folder_list, data_list)
    ]
    with Pool(32) as p:
        results = list(tqdm(p.imap(check_data_exists, data_dict), total=len(data_dict)))
    not_exists = []
    for data_path in results:
        not_exists.extend(data_path)

    # Write not exists into txt files
    with open("not_exists.txt", "w") as f:
        for data_path in not_exists:
            f.write(data_path + "\n")
