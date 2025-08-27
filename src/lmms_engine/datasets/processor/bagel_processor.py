from .processor import Processor
from PIL import Image
from lmms_engine.mapping_func import register_processor
from .config import ProcessorConfig
from transformers import Qwen2Tokenizer
from dacite import from_dict
import numpy

def add_special_tokens(tokenizer):
    all_special_tokens = []
    for k, v in tokenizer.special_tokens_map.items():
        if isinstance(v, str):
            all_special_tokens.append(v)
        elif isinstance(v, list):
            all_special_tokens += v

    new_tokens = []

    if '<|im_start|>' not in all_special_tokens:
        new_tokens.append('<|im_start|>')

    if '<|im_end|>' not in all_special_tokens:
        new_tokens.append('<|im_end|>')

    if '<|vision_start|>' not in all_special_tokens:
        new_tokens.append('<|vision_start|>')

    if '<|vision_end|>' not in all_special_tokens:
        new_tokens.append('<|vision_end|>')

    num_new_tokens = tokenizer.add_tokens(new_tokens)
    bos_token_id = tokenizer.convert_tokens_to_ids('<|im_start|>')
    eos_token_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
    start_of_image = tokenizer.convert_tokens_to_ids('<|vision_start|>')
    end_of_image = tokenizer.convert_tokens_to_ids('<|vision_end|>')

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

    def build(self):
        self.tokenizer = self._build_processor()

    def _build_processor(self):
        tokenizer = Qwen2Tokenizer.from_pretrained(self.config.processor_name)
        tokenizer, _, _ = add_special_tokens(tokenizer)
        return tokenizer

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
        