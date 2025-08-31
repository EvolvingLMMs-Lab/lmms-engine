import json

import numpy
from dacite import from_dict
from PIL import Image
from transformers import Qwen2Tokenizer

from lmms_engine.mapping_func import register_processor

from .bagel_utils.transforms import ImageTransform
from .bagel_utils.video_utils import FrameSampler
from .config import ProcessorConfig
from .processor import Processor


def pil_img2rgb(image):
    if image.mode == "RGBA" or image.info.get("transparency", None) is not None:
        image = image.convert("RGBA")
        white = Image.new(mode="RGB", size=image.size, color=(255, 255, 255))
        white.paste(image, mask=image.split()[3])
        image = white
    else:
        image = image.convert("RGB")

    return image


def add_special_tokens(tokenizer):
    all_special_tokens = []
    for k, v in tokenizer.special_tokens_map.items():
        if isinstance(v, str):
            all_special_tokens.append(v)
        elif isinstance(v, list):
            all_special_tokens += v

    new_tokens = []

    if "<|im_start|>" not in all_special_tokens:
        new_tokens.append("<|im_start|>")

    if "<|im_end|>" not in all_special_tokens:
        new_tokens.append("<|im_end|>")

    if "<|vision_start|>" not in all_special_tokens:
        new_tokens.append("<|vision_start|>")

    if "<|vision_end|>" not in all_special_tokens:
        new_tokens.append("<|vision_end|>")

    num_new_tokens = tokenizer.add_tokens(new_tokens)
    bos_token_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    start_of_image = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    end_of_image = tokenizer.convert_tokens_to_ids("<|vision_end|>")

    new_token_ids = dict(
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        start_of_image=start_of_image,
        end_of_image=end_of_image,
    )

    return tokenizer, new_token_ids, num_new_tokens


@register_processor("bagel")
class BagelProcessor(Processor):
    def __init__(self, config: ProcessorConfig | dict) -> None:
        if isinstance(config, dict):
            config = from_dict(ProcessorConfig, config)
        self.config = config
        self.dataset_args = self.config.kwargs if self.config.kwargs else {}

    def build(self):
        self.tokenizer = self._build_processor()

        if "frame_sampler_args" in self.dataset_args.keys():
            self.frame_sampler = FrameSampler(
                **self.dataset_args.pop("frame_sampler_args")
            )
        if "image_transform_args" in self.dataset_args.keys():
            self.transform = ImageTransform(
                **self.dataset_args.pop("image_transform_args")
            )
            self.transform_stride = self.transform.stride
        if "vit_image_transform_args" in self.dataset_args.keys():
            self.vit_transform = ImageTransform(
                **self.dataset_args.pop("vit_image_transform_args")
            )

    def _build_processor(self):
        tokenizer = Qwen2Tokenizer.from_pretrained(self.config.processor_name)
        tokenizer, _, _ = add_special_tokens(tokenizer)
        return tokenizer

    def _process_t2i(self, user_text, user_images, result_image):
        assert not user_images, "current we do not support user images"
        num_tokens = 0
        image_tensor = self.transform(result_image)
        height, width = image_tensor.shape[1:]
        num_tokens += width * height // self.transform_stride**2
        caption_token = self.tokenizer.encode(user_text)
        sequence_plan, text_ids_list = [], []
        text_ids = caption_token
        num_tokens += len(caption_token)
        text_ids_list.append(text_ids)
        sequence_plan.append(
            {
                "type": "text",
                "enable_cfg": 1,
                "loss": 0,
                "special_token_loss": 0,
                "special_token_label": None,
            }
        )

        sequence_plan.append(
            {
                "type": "vae_image",
                "enable_cfg": 0,
                "loss": 1,
                "special_token_loss": 0,
                "special_token_label": None,
            }
        )

        sample = {
            "image_tensor_list": [image_tensor],
            "text_ids_list": text_ids_list,
            "num_tokens": num_tokens,
            "sequence_plan": sequence_plan,
            # "data_indexes": {
            #     "data_indexes": [parquet_idx, row_group_id, row_idx],
            #     "worker_id": worker_id,
            #     "dataset_name": self.dataset_name,
            # }
        }

        return sample

    def _process_t2t(self, user_text, user_images, assistant_text):
        raise NotImplementedError(
            "BagelProcessor does not support text-to-text generation"
        )

    def process(
        self,
        images: list[Image.Image],
        hf_messages,
        audios: list[numpy.ndarray] | None = None,
        sampling_rate: int | None = None,
        videos=None,
        add_system_prompt=True,
    ):
        if audios or videos:
            raise ValueError("Audios and videos are not supported for BagelProcessor")

        if len(hf_messages) != 2:
            raise ValueError("BagelProcessor only supports two-turn conversations")

        text = {
            "user": [],
            "assistant": [],
        }
        processed_images = {
            "user": [],
            "assistant": [],
        }

        current_image_ptr = 0

        for message in hf_messages:
            role = message["role"]
            for content in message["content"]:
                if content["type"] == "text":
                    text[role].append(content["text"])
                elif content["type"] == "image":
                    processed_images[role].append(
                        pil_img2rgb(images[current_image_ptr])
                    )
                    current_image_ptr += 1

        user_text = "\n".join(text["user"])
        assistant_text = "\n".join(text["assistant"])
        user_images = processed_images["user"]
        assistant_images = processed_images["assistant"]

        if assistant_images:
            assert (
                len(assistant_text) == 0
            ), "BagelProcessor only supports image-to-image generation, but assistant text is provided"
            assert (
                len(assistant_images) == 1
            ), "BagelProcessor only supports one assistant image"

            return self._process_t2i(user_text, user_images, assistant_images[0])
        else:
            return self._process_t2t(user_text, user_images, assistant_text)
