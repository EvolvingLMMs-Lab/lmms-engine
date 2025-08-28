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

        dataset_args = self.config.kwargs

        if "frame_sampler_args" in dataset_args.keys():
            frame_sampler = FrameSampler(**dataset_args.pop("frame_sampler_args"))
            dataset_args["frame_sampler"] = frame_sampler
        if "image_transform_args" in dataset_args.keys():
            transform = ImageTransform(**dataset_args.pop("image_transform_args"))
            dataset_args["transform"] = transform
        if "vit_image_transform_args" in dataset_args.keys():
            vit_transform = ImageTransform(
                **dataset_args.pop("vit_image_transform_args")
            )
            dataset_args["vit_transform"] = vit_transform

    def build(self):
        self.tokenizer = self._build_processor()

    def _build_processor(self):
        tokenizer = Qwen2Tokenizer.from_pretrained(self.config.processor_name)
        tokenizer, _, _ = add_special_tokens(tokenizer)
        return tokenizer

    def _process_t2i(self, user_text, user_images, result_image):
        pass

    def _process_t2t(self, user_text, user_images, assistant_text):
        pass

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
