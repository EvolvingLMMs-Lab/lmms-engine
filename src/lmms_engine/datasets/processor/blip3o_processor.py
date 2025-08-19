from dataclasses import dataclass
import pyarrow.parquet as pq
import torch
import transformers
import yaml
from PIL import Image, ImageFile
from lmms_engine.utils.logging_utils import Logging
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transforms
from datasets import load_dataset, concatenate_datasets
from .processor import Processor
# from blip3o.utils import rank0_print

@dataclass
class Blip3oConstants:
    IGNORE_INDEX: int = -100
    IMAGE_TOKEN_INDEX: int = -200
    DEFAULT_IMAGE_TOKEN: str = "<image>"
    DEFAULT_IMAGE_PATCH_TOKEN: str = "<im_patch>"
    DEFAULT_IM_START_TOKEN: str = "<im_start>"
    DEFAULT_IM_END_TOKEN: str = "<im_end>"


ImageFile.LOAD_TRUNCATED_IMAGES = True

target_transform = transforms.Compose(
    [
        transforms.Resize(1024),
        transforms.CenterCrop(1024),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize([0.5], [0.5]),
    ]
)


def expand2square(pil_img, background_color):
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


def preprocess_multimodal(sources: list[str], data_args) -> dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            replace_token = Blip3oConstants.DEFAULT_IMAGE_TOKEN
            # NOTE: only add im_start_end when image generation
            if data_args.mm_use_im_start_end and sentence['from'] == 'gpt':
                replace_token = Blip3oConstants.DEFAULT_IM_START_TOKEN + replace_token + Blip3oConstants.DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(Blip3oConstants.DEFAULT_IMAGE_TOKEN, replace_token)

            # For videoInstruct-100k noisy_data. TODO: Ask Yuanhan to clean the data instead of leaving the noise code here.
            sentence["value"] = sentence["value"].replace("QA_GT_caption_based_noisy", "")

    return sources


def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.") -> dict:
    # roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}
    roles = {"human": "user", "gpt": "assistant"}

    #tokenizer = copy.deepcopy(tokenizer)
    # When there is actually an image, we add the image tokens as a special token
    if 'image_token_index' not in globals():
        tokenizer.add_tokens(["<image>"], special_tokens=True)
        global image_token_index
        image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    # if has_image:
    #     tokenizer.add_tokens(["<image>"], special_tokens=True)

    # image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    im_start, im_end = tokenizer.additional_special_tokens_ids[:2]
    # unmask_tokens = ["<|im_start|>", "<|im_start|>", "\n"]
    unmask_tokens_idx =  [198, im_start, im_end]
    # nl_tokens = tokenizer("\n").input_ids

    # Reset Qwen chat templates so that it won't include system message every time we apply
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    # _system = tokenizer("system").input_ids + nl_tokens
    # _user = tokenizer("user").input_ids + nl_tokens
    # _assistant = tokenizer("assistant").input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        # New version, use apply chat template
        # Build system message for each sentence
        input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])


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

            role =  roles.get(role, role)
            
            conv = [{"role" : role, "content" : content}]
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

    return dict(
        input_ids=input_ids,  
        labels=targets,  
    )

class Blip3oProcessor(Processor):
    pass
