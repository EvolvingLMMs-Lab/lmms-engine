# Based on https://github.com/JiuhaiChen/BLIP3o/blob/BLIP3o-NEXT/blip3o/data/dataset.py

import torch
import transformers
from PIL import Image, ImageFile
from torchvision.transforms import v2 as transforms
from transformers import AutoTokenizer

# from blip3o.utils import rank0_print
from lmms_engine.models.blip3o.constants import Blip3oConstants

from .config import ProcessorConfig
from .processor import Processor

ImageFile.LOAD_TRUNCATED_IMAGES = True


def expand2square(pil_img: Image.Image, background_color) -> Image.Image:
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def preprocess_multimodal(sources: list[str], data_args):
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            replace_token = Blip3oConstants.DEFAULT_IMAGE_TOKEN
            # NOTE: only add im_start_end when image generation
            if data_args.mm_use_im_start_end and sentence["from"] == "gpt":
                replace_token = (
                    Blip3oConstants.DEFAULT_IM_START_TOKEN
                    + replace_token
                    + Blip3oConstants.DEFAULT_IM_END_TOKEN
                )
            sentence["value"] = sentence["value"].replace(
                Blip3oConstants.DEFAULT_IMAGE_TOKEN, replace_token
            )

            # For videoInstruct-100k noisy_data. TODO: Ask Yuanhan to clean the data instead of leaving the noise code here.
            sentence["value"] = sentence["value"].replace(
                "QA_GT_caption_based_noisy", ""
            )

    return sources


def preprocess_qwen(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    max_len=2048,
    system_message: str = "You are a helpful assistant.",
) -> dict:
    # roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}
    roles = {"human": "user", "gpt": "assistant"}

    # tokenizer = copy.deepcopy(tokenizer)
    # When there is actually an image, we add the image tokens as a special token
    if "image_token_index" not in globals():
        tokenizer.add_tokens(["<image>"], special_tokens=True)
        global image_token_index
        image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    # if has_image:
    #     tokenizer.add_tokens(["<image>"], special_tokens=True)

    # image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    im_start, im_end = tokenizer.additional_special_tokens_ids[:2]
    # unmask_tokens = ["<|im_start|>", "<|im_start|>", "\n"]
    unmask_tokens_idx = [198, im_start, im_end]
    # nl_tokens = tokenizer("\n").input_ids

    # Reset Qwen chat templates so that it won't include system message every time we apply
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    # _system = tokenizer("system").input_ids + nl_tokens
    # _user = tokenizer("user").input_ids + nl_tokens
    # _assistant = tokenizer("assistant").input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []
    for source in sources:
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        # New version, use apply chat template
        # Build system message for each sentence
        input_id += tokenizer.apply_chat_template(
            [{"role": "system", "content": system_message}]
        )

        # target += [IGNORE_INDEX] * len(input_id)
        target += input_id

        for conv in source:
            # Make sure blip3o data can load
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role = roles.get(role, role)

            conv = [{"role": role, "content": content}]
            encode_id = tokenizer.apply_chat_template(conv)
            input_id += encode_id
            if role in ["user", "system"]:
                # target += [IGNORE_INDEX] * len(encode_id)
                target += encode_id

            else:
                target += encode_id

        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            if encode_id == image_token_index:
                input_id[idx] = Blip3oConstants.IMAGE_TOKEN_INDEX
        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return {
        "input_ids": input_ids,
        "labels": targets,
    }


class Blip3oProcessor(Processor):
    def __init__(self, config: ProcessorConfig):
        self.config = config
        self.target_transform = transforms.Compose(
            [
                transforms.Resize(1024),
                transforms.CenterCrop(1024),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.processor_name, padding_side="right"
        )
        if self.tokenizer.unk_token is not None:
            self.tokenizer.pad_token = self.tokenizer.unk_token

    def process(self):
        sources = self.list_data_dict[i]

        if sources["type"] == "T2I":
            sources["conversations"] = [
                {
                    "from": "human",
                    "value": f"Please generate image based on the following caption: {sources['txt']}",
                },
                {"from": "gpt", "value": "<image>"},
            ]

        elif sources["type"] == "I2I":
            sources["conversations"] = [
                {
                    "from": "human",
                    "value": f"<image>\nPlease reconstruct the given image.",
                },
                {"from": "gpt", "value": ""},
            ]

        else:
            raise ValueError(
                "Unknown source type. Please check the 'type' in 'sources'."
            )

        if "image" in sources:
            if sources["type"] == "T2I" or sources["type"] == "I2I":
                image_files = self.list_data_dict[i]["image"]

            if not isinstance(image_files, list):
                image_files = [image_files]

            images = []

            for img in image_files:
                try:
                    if sources["type"] == "T2I" or sources["type"] == "I2I":
                        img = img.convert("RGB")
                    else:
                        raise ValueError(
                            "Unknown source type. Please check the 'type' in 'sources'."
                        )
                    images.append(img)
                except Exception as e:
                    print(f"Error opening image {img}: {e}")
                    images = None
                    break  # Skip to the next image if there's an error

            ## test if can apply img_process
            if not images is None:
                try:
                    process_images = [self.process_image(f) for f in images]
                except Exception as e:
                    print(f"Error wrong number of channels: {e}")
                    images = None

            # If no valid images were found, randomly pick another item
            if images is None:
                print(sources)
                print(f"warning false image!!!!!!")
                i = random.randint(0, len(self.list_data_dict) - 1)
                continue

            sources = preprocess_multimodal(
                copy.deepcopy([sources["conversations"]]), self.data_args
            )
        else:
            sources = copy.deepcopy([sources["conversations"]])

        data_dict = preprocess_qwen(
            sources, self.tokenizer, has_image=("image" in self.list_data_dict[i])
        )
        if isinstance(i, int):
            data_dict = dict(
                input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0]
            )

        # image exist in the data
        if "image" in self.list_data_dict[i]:
            data_dict["image"] = process_images
            data_dict["target_image"] = [self.process_target_image(f) for f in images]

        data_dict["ids"] = (
            self.list_data_dict[i]["id"] if "id" in self.list_data_dict[i] else "unk"
        )
        return data_dict
