"""
EditReward Scorer - A single-file implementation for image editing quality evaluation.

This module provides:
- EditRewardScorer: Main scorer class for evaluating edited images
- Supports both local inference and remote API calls

Usage:
    from lmms_engine.utils.rewards.editreward_scorer import EditRewardScorer
    
    scorer = EditRewardScorer(
        checkpoint_path=checkpoint_path,
        model_name_or_path="MiMo-VL-7B-SFT-2508",
        device="cuda",
        reward_dim="overall_detail",
        # These match config: EditReward-MiMo-VL-7B-SFT-2508.yaml
        rm_head_type="ranknet_multi_head",  # rm_head_type: ranknet_multi_head
        pooling_strategy="mean",             # pooling_strategy: mean
        use_special_tokens=True,             # use_special_tokens: true
        output_dim=2,                        # output_dim: 2 (for uncertainty loss)
        rm_head_kwargs=None,                 # Use default structure (hidden->1024->16->output)
    )

    scores = scorer.score(
        prompts=["Add a cat to the image"],
        source_images=[source_pil_image],
        edited_images=[edited_pil_image]
    )
"""

import base64
import math
import os
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import requests
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

try:
    import safetensors.torch
except ImportError:
    safetensors = None

try:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from trl import get_kbit_device_map, get_quantization_config
except ImportError as e:
    raise ImportError(f"Required packages not found: {e}. Please install transformers, peft, trl")

try:
    import flash_attn

    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

# ============================================================================
# Prompts for EditReward evaluation
# ============================================================================

INSTRUCTION_EDIT_FOLLOWING = """
You are tasked with evaluating an edited image **in comparison with the original source image** based on **Instruction Following & Semantic Fidelity**, and assigning a score from 1 to 4, with 1 being the worst and 4 being the best.  
This dimension focuses on how accurately, completely, and exclusively the model executed the given text instruction.

**Inputs Provided:**  
- Source Image (before editing)  
- Edited Image (after applying the instruction)  
- Text Instruction  

**Sub-Dimensions to Evaluate:**

- **Semantic Accuracy:**  
  Assess whether the edited content accurately captures the semantics of the instruction. The edited result should precisely match the intended meaning. For example, if the instruction is "replace apples with oranges," the object must clearly be oranges, not other fruits.

- **Completeness of Editing:**  
  Check whether **all parts** of the instruction are fully executed. For multi-step edits (e.g., "replace a red car with a blue bicycle"), both the color change and the object replacement must be done without omissions.

- **Exclusivity of Edit (No Over-Editing):**  
  Ensure that only the requested parts are changed. The rest of the image (as seen in the source) should remain unaltered. For example, if the instruction only involves replacing an object, the background, lighting, and unrelated objects should not be unnecessarily modified.

**Scoring Criteria:**
- **4 (Very Good):** Perfectly accurate, complete, and exclusive execution of the instruction.  
- **3 (Relatively Good):** Largely correct, but with minor omissions or slight over-editing.  
- **2 (Relatively Poor):** Major misinterpretation, incomplete edits, or noticeable unintended changes.  
- **1 (Very Poor):** Instruction ignored or completely wrong execution.

Text instruction - {text_prompt}
"""

INSTRUCTION_EDIT_QUALITY = """
You are tasked with evaluating an edited image **in comparison with the original source image** based on **Visual Quality & Realism**, and assigning a score from 1 to 4, with 1 being the worst and 4 being the best.  
This dimension focuses on how realistic, artifact-free, and aesthetically appealing the edited image is, while remaining consistent with the source image.

**Inputs Provided:**  
- Source Image (before editing)  
- Edited Image (after applying the instruction)  
- Text Instruction  

**Sub-Dimensions to Evaluate:**

- **Plausibility & Physical Consistency:**  
  Check whether the edit aligns with the laws of physics and the scene context. Lighting, shadows, reflections, perspective, size, and interactions with the environment should all appear natural compared to the source image.

- **Artifact-Free Quality:**  
  Look for technical flaws such as blur, distortions, pixel misalignment, unnatural textures, or seams around edited regions. High-quality results should be free from such visible artifacts.

- **Aesthetic Quality:**  
  Evaluate the overall harmony and visual appeal. The image should look natural, balanced, and pleasant. Colors, composition, and atmosphere should enhance the image rather than degrade it.

**Scoring Criteria:**
- **4 (Very Good):** Perfectly realistic, artifact-free, seamless, and aesthetically pleasing.  
- **3 (Relatively Good):** Mostly realistic and clean, with only minor flaws that do not significantly distract.  
- **2 (Relatively Poor):** Noticeable physical inconsistencies or visible artifacts that make the edit unnatural.  
- **1 (Very Poor):** Severe artifacts, incoherent composition, or visually unusable result.

Text instruction - {text_prompt}
"""


INSTRUCTION_EDIT_OVERALL_DETAILED = """
You are tasked with evaluating an edited image **in comparison with the original source image**, and assigning a score from 1 to 8, with 1 being the worst and 8 being the best.  
This score should reflect **both how accurately the instruction was followed and the visual quality of the edited image**.

**Inputs Provided:**  
- Source Image (before editing)  
- Edited Image (after applying the instruction)  
- Text Instruction  

**Dimension 1: Instruction Following & Semantic Fidelity**  
Evaluate how well the edited image follows the given instruction. Consider the following sub-dimensions:
- **Semantic Accuracy:** Check if the edited content accurately captures the intended meaning of the instruction. For example, if the instruction is "replace apples with oranges," the object must clearly be oranges, not other fruits.
- **Completeness of Editing:** Verify that all aspects of the instruction are fully executed. Multi-step edits should be completely applied without omissions.
- **Exclusivity of Edit (No Over-Editing):** Ensure that only the requested changes are applied; the rest of the image should remain consistent with the source image without unintended modifications.

**Dimension 2: Visual Quality & Realism**  
Evaluate the realism, technical quality, and aesthetic appeal of the edited image. Consider the following sub-dimensions:
- **Plausibility & Physical Consistency:** Check whether the edit aligns with natural laws and scene context (lighting, shadows, reflections, perspective, and object interactions).
- **Artifact-Free Quality:** Assess for technical flaws such as blur, distortions, pixel misalignment, unnatural textures, or seams around edited regions.
- **Aesthetic Quality:** Consider overall harmony and visual appeal. Colors, composition, atmosphere, and balance should enhance the image without degrading realism.

**Scoring Criteria (1–8):**  
- **8 (Very Good):** Perfect instruction following and flawless visual quality; edits are accurate, complete, exclusive, and visually seamless.  
- **7 (Relatively Good):** Very good instruction following and high visual quality; minor, non-distracting flaws.  
- **6 (Good):** Good instruction following or mostly good visual quality; minor omissions or slight artifacts.  
- **5 (Moderate):** Partially correct edits or moderate visual issues; noticeable flaws but understandable.  
- **4 (Relatively Poor):** Significant misinterpretation, incomplete edits, or noticeable visual artifacts.  
- **3 (Poor):** Major errors in instruction following and/or poor visual quality; hard to fully understand.  
- **2 (Very Poor):** Very poor edits with large semantic errors and strong visual artifacts.  
- **1 (Failed):** Completely wrong edits or visually unusable result.

Text instruction - {text_prompt}
"""


PROMPT_WITH_SPECIAL_TOKEN = """
Please provide the overall ratings of this image: <|Reward|>

END
"""

PROMPT_WITHOUT_SPECIAL_TOKEN = """
Please provide the overall ratings of this image: 
"""

# ============================================================================
# Image Processing Utilities
# ============================================================================

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200


def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor


def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> Tuple[int, int]:
    """Rescale image maintaining aspect ratio within pixel constraints."""
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(f"Aspect ratio must be smaller than {MAX_RATIO}")
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def fetch_image(ele: dict, size_factor: int = IMAGE_FACTOR) -> Image.Image:
    """Fetch and resize image from various sources."""
    image = ele.get("image") or ele.get("image_url")
    image_obj = None

    if isinstance(image, Image.Image):
        image_obj = image
    elif isinstance(image, torch.Tensor):
        # Convert tensor to PIL Image
        if image.dim() == 4:
            image = image[0]
        if image.shape[0] in [1, 3]:
            image = image.permute(1, 2, 0)
        image_np = (image.cpu().numpy() * 255).astype(np.uint8)
        image_obj = Image.fromarray(image_np)
    elif isinstance(image, str):
        if image.startswith(("http://", "https://")):
            image_obj = Image.open(requests.get(image, stream=True).raw)
        elif image.startswith("file://"):
            image_obj = Image.open(image[7:])
        elif image.startswith("data:image"):
            if "base64," in image:
                _, base64_data = image.split("base64,", 1)
                data = base64.b64decode(base64_data)
                image_obj = Image.open(BytesIO(data))
        else:
            image_obj = Image.open(image)

    if image_obj is None:
        raise ValueError(f"Unrecognized image input: {type(image)}")

    if isinstance(image_obj, Image.Image):
        image_obj = image_obj.convert("RGB")
        width, height = image_obj.size
        min_pixels = ele.get("min_pixels", MIN_PIXELS)
        max_pixels = ele.get("max_pixels", MAX_PIXELS)
        resized_height, resized_width = smart_resize(
            height, width, factor=size_factor, min_pixels=min_pixels, max_pixels=max_pixels
        )
        image_obj = image_obj.resize((resized_width, resized_height), Image.BICUBIC)

    return image_obj


def process_vision_info(conversations: List[dict]) -> Tuple[List[Image.Image], None]:
    """Extract and process images from conversation format."""
    vision_infos = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]

    for conversation in conversations:
        for message in conversation:
            if isinstance(message.get("content"), list):
                for ele in message["content"]:
                    if "image" in ele or "image_url" in ele or ele.get("type") in ("image", "image_url"):
                        vision_infos.append(ele)

    image_inputs = [fetch_image(info) for info in vision_infos]
    return image_inputs if image_inputs else None, None


# ============================================================================
# Model Definition
# ============================================================================


class EditRewardModel(Qwen2_5_VLForConditionalGeneration):
    """Qwen2.5-VL based reward model with multi-head support."""

    def __init__(
        self,
        config,
        output_dim: int = 4,
        reward_token: str = "last",
        special_token_ids: Optional[List[int]] = None,
        rm_head_type: str = "ranknet_multi_head",
        rm_head_kwargs: Optional[dict] = None,
        pooling_strategy: str = "min",
    ):
        super().__init__(config)
        self.output_dim = output_dim
        self.rm_head_type = rm_head_type
        self.reward_token = reward_token
        self.special_token_ids = special_token_ids
        self.pooling_strategy = pooling_strategy

        self.rm_head = None
        self.rm_heads = None

        if rm_head_type == "default":
            self.rm_head = nn.Linear(config.hidden_size, output_dim, bias=False)
        elif rm_head_type in ("ranknet", "ranknet_share_head"):
            self.rm_head = self._build_head(config.hidden_size, output_dim, rm_head_kwargs)
        elif rm_head_type in ("ranknet_multi_head", "ranknet_multi_head_regression"):
            num_heads = rm_head_kwargs.get("num_heads", 2) if rm_head_kwargs else 2
            self.rm_heads = nn.ModuleDict(
                {
                    f"head_{i}": self._build_head(config.hidden_size, output_dim, rm_head_kwargs)
                    for i in range(num_heads)
                }
            )

        # Set dtype for rm_head(s)
        if self.rm_head is not None:
            self.rm_head.to(torch.float32)
        if self.rm_heads is not None:
            for h in self.rm_heads.values():
                h.to(torch.float32)

        if self.special_token_ids is not None:
            self.reward_token = "special"

    def _build_head(self, hidden_size: int, output_dim: int, kwargs: Optional[dict]) -> nn.Module:
        """Build a reward head matching the original EditReward implementation."""
        if kwargs is not None:
            layers = []
            num_layers = kwargs.get("num_layers", 3)
            head_hidden = kwargs.get("hidden_size", 1024)
            dropout = kwargs.get("dropout", 0.1)

            for layer in range(num_layers):
                if layer == 0:
                    layers += [
                        nn.Linear(hidden_size, head_hidden),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                    ]
                elif layer < num_layers - 1:
                    layers += [
                        nn.Linear(head_hidden, head_hidden),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                    ]
                else:
                    layers.append(nn.Linear(head_hidden, output_dim, bias=kwargs.get("bias", False)))
            return nn.Sequential(*layers)
        else:
            # Default structure: hidden_size -> 1024 -> 16 -> output_dim
            # This matches the checkpoint trained without custom rm_head_kwargs
            return nn.Sequential(
                nn.Linear(hidden_size, 1024),
                nn.ReLU(),
                nn.Dropout(0.05),
                nn.Linear(1024, 16),
                nn.ReLU(),
                nn.Linear(16, output_dim),
            )

    def _pool_logits(self, logits: torch.Tensor, input_ids: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Pool logits based on reward_token strategy."""
        if self.reward_token == "last":
            if self.config.pad_token_id is None:
                sequence_lengths = -1
            else:
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            return logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
        elif self.reward_token == "mean":
            if self.config.pad_token_id is None:
                return logits.mean(dim=1)
            else:
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                return torch.stack([logits[i, : sequence_lengths[i]].mean(dim=0) for i in range(batch_size)])
        elif self.reward_token == "special":
            special_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for token_id in self.special_token_ids:
                special_token_mask = special_token_mask | (input_ids == token_id)
            pooled = logits[special_token_mask, ...]
            return pooled.view(batch_size, -1)
        else:
            raise ValueError(f"Invalid reward_token: {self.reward_token}")

    def _forward_single_batch(self, batch_dict: dict, head_module: nn.Module) -> torch.Tensor:
        """Forward a single batch through model and head."""
        input_ids = batch_dict.get("input_ids")
        attention_mask = batch_dict.get("attention_mask")
        pixel_values = batch_dict.get("pixel_values")
        image_grid_thw = batch_dict.get("image_grid_thw")

        inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_embeds = (
                self.visual(pixel_values, grid_thw=image_grid_thw)
                if image_grid_thw is not None
                else self.visual(pixel_values)
            )
            image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)

        outputs = self.model(
            input_ids=None, inputs_embeds=inputs_embeds, attention_mask=attention_mask, return_dict=True
        )
        hidden_states = outputs[0]
        batch_size = input_ids.shape[0]

        with torch.autocast(device_type="cuda", dtype=torch.float32):
            logits = head_module(hidden_states)

        return self._pool_logits(logits, input_ids, batch_size)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        batch_dim1: Optional[dict] = None,
        batch_dim2: Optional[dict] = None,
        **kwargs,
    ):
        # Single head logic
        if self.rm_head_type in ("ranknet", "default"):
            inputs_embeds = self.get_input_embeddings()(input_ids)
            if pixel_values is not None:
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds.to(inputs_embeds))

            outputs = self.model(
                input_ids=None, inputs_embeds=inputs_embeds, attention_mask=attention_mask, return_dict=True
            )
            hidden_states = outputs[0]
            batch_size = input_ids.shape[0]

            with torch.autocast(device_type="cuda", dtype=torch.float32):
                logits = self.rm_head(hidden_states)

            pooled = self._pool_logits(logits, input_ids, batch_size)
            return {"logits": pooled}

        # Multi-head logic
        elif self.rm_head_type in ("ranknet_multi_head", "ranknet_multi_head_regression", "ranknet_share_head"):
            num_heads = len(self.rm_heads) if self.rm_heads else 2
            head_batches = {f"batch_dim{i+1}": kwargs.get(f"batch_dim{i+1}") for i in range(num_heads)}
            if batch_dim1 is not None:
                head_batches["batch_dim1"] = batch_dim1
            if batch_dim2 is not None:
                head_batches["batch_dim2"] = batch_dim2

            logits_per_head = []
            for i in range(num_heads):
                batch = head_batches.get(f"batch_dim{i+1}")
                if batch is not None:
                    head = self.rm_heads[f"head_{i}"] if self.rm_heads else self.rm_head
                    logits_per_head.append(self._forward_single_batch(batch, head))

            if self.pooling_strategy is None:
                return logits_per_head

            stacked = torch.stack(logits_per_head, dim=0)
            if self.pooling_strategy == "min":
                final_logits = stacked.min(dim=0).values
            elif self.pooling_strategy == "mean":
                final_logits = stacked.mean(dim=0)
            else:
                final_logits = stacked.mean(dim=0)

            return {"logits": final_logits}


# ============================================================================
# Configuration Classes
# ============================================================================


@dataclass
class ModelConfig:
    model_name_or_path: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    model_revision: str = "main"
    rm_head_type: str = "ranknet_multi_head"
    rm_head_kwargs: Optional[dict] = None
    pooling_strategy: str = "min"
    output_dim: int = 1
    use_special_tokens: bool = False
    torch_dtype: Optional[str] = "bfloat16"


@dataclass
class PEFTConfig:
    lora_enable: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    num_lora_modules: int = -1
    lora_namespan_exclude: Optional[List[str]] = None
    lora_modules_to_save: Optional[List[str]] = None


# ============================================================================
# Main Scorer Class
# ============================================================================


class EditRewardScorer:
    """
    EditReward scorer for evaluating image editing quality.

    Args:
        checkpoint_path: Path to model checkpoint
        model_name_or_path: Base model name (default: MiMo-VL-7B-SFT-2508)
        device: Device to run on (default: cuda)
        reward_dim: Dimension to evaluate (dim1=instruction following, dim2=visual quality)
        rm_head_type: Type of reward head (default: ranknet_multi_head)
        use_bf16: Whether to use bfloat16 (default: True)
        use_special_tokens: Whether to use special reward token (default: True)
        pooling_strategy: How to pool multi-head outputs (default: mean)
        output_dim: Output dimension for reward head (default: 2 for uncertainty)
        rm_head_kwargs: Custom rm_head config. If None, uses default structure
                       (hidden_size -> 1024 -> 16 -> output_dim)
    """

    def __init__(
        self,
        checkpoint_path: str,
        model_name_or_path: str,
        device: str = "cuda",
        reward_dim: str = "dim1",
        rm_head_type: str = "ranknet_multi_head",
        use_bf16: bool = True,
        use_special_tokens: bool = True,
        pooling_strategy: str = "mean",
        output_dim: int = 2,
        rm_head_kwargs: Optional[dict] = None,
    ):
        self.device = device
        self.reward_dim = reward_dim
        self.rm_head_type = rm_head_type
        self.use_special_tokens = use_special_tokens

        # rm_head_kwargs=None means use default structure: hidden->1024->16->output_dim
        # This matches checkpoints trained without custom rm_head_kwargs

        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_name_or_path, padding_side="right")

        # Handle special tokens
        special_token_ids = None
        if use_special_tokens:
            special_tokens = ["<|Reward|>"]
            self.processor.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
            special_token_ids = self.processor.tokenizer.convert_tokens_to_ids(special_tokens)

        # Create model
        self.model = EditRewardModel.from_pretrained(
            model_name_or_path,
            output_dim=output_dim,  # Use configurable output_dim
            reward_token="special" if use_special_tokens else "last",
            special_token_ids=special_token_ids,
            torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
            attn_implementation="flash_attention_2" if HAS_FLASH_ATTN else "sdpa",
            rm_head_type=rm_head_type,
            rm_head_kwargs=rm_head_kwargs,
            pooling_strategy=pooling_strategy,
        )

        if use_special_tokens:
            self.model.resize_token_embeddings(len(self.processor.tokenizer))

        # Load checkpoint
        self._load_checkpoint(checkpoint_path)

        # Move to device and set dtype
        if use_bf16:
            self.model.to(torch.bfloat16)

        # Keep rm_heads in float32 for numerical stability
        if self.model.rm_heads is not None:
            for h in self.model.rm_heads.values():
                h.to(torch.float32)
        elif self.model.rm_head is not None:
            self.model.rm_head.to(torch.float32)

        self.model.to(device)
        self.model.eval()

    def _load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        full_ckpt = os.path.join(checkpoint_path, "model.pth")
        full_ckpt_safetensors = os.path.join(checkpoint_path, "model.safetensors")

        if os.path.exists(full_ckpt):
            state_dict = torch.load(full_ckpt, map_location="cpu")
        elif os.path.exists(full_ckpt_safetensors):
            if safetensors is None:
                raise ImportError("safetensors is required to load .safetensors files")
            state_dict = safetensors.torch.load_file(full_ckpt_safetensors, device="cpu")
        else:
            raise ValueError(f"Checkpoint not found at {checkpoint_path}")

        if "model" in state_dict:
            state_dict = state_dict["model"]

        self.model.load_state_dict(state_dict, strict=True)

    def _prepare_input(self, data):
        """Move inputs to device."""
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            return data.to(device=self.device)
        return data

    def _build_messages(
        self,
        prompts: List[str],
        source_images: List,
        edited_images: List,
        reward_dim: str,
        max_pixels: int = 256 * 28 * 28,
        min_pixels: int = 256 * 28 * 28,
    ) -> List:
        """Build conversation messages for the model."""
        message_list = []

        for text, src_img, edit_img in zip(prompts, source_images, edited_images):
            if reward_dim == "dim1":
                base_prompt = INSTRUCTION_EDIT_FOLLOWING.format(text_prompt=text)
            elif reward_dim == "dim2":
                base_prompt = INSTRUCTION_EDIT_QUALITY.format(text_prompt=text)
            elif reward_dim == "overall_detail":
                base_prompt = INSTRUCTION_EDIT_OVERALL_DETAILED.format(text_prompt=text)
            else:
                base_prompt = INSTRUCTION_EDIT_FOLLOWING.format(text_prompt=text)

            final_text = base_prompt + (
                PROMPT_WITH_SPECIAL_TOKEN if self.use_special_tokens else PROMPT_WITHOUT_SPECIAL_TOKEN
            )

            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": src_img, "min_pixels": min_pixels, "max_pixels": max_pixels},
                        {"type": "image", "image": edit_img, "min_pixels": max_pixels, "max_pixels": max_pixels},
                        {"type": "text", "text": final_text},
                    ],
                }
            ]
            message_list.append(message)

        return message_list

    def _build_batch(
        self,
        prompts: List[str],
        source_images: List,
        edited_images: List,
        reward_dim: str,
    ) -> dict:
        """Build a batch for model inference."""
        messages = self._build_messages(prompts, source_images, edited_images, reward_dim)
        image_inputs, _ = process_vision_info(messages)

        batch = self.processor(
            text=self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        return self._prepare_input(batch)

    @torch.no_grad()
    def score(
        self,
        prompts: List[str],
        source_images: List[Union[str, Image.Image, torch.Tensor]],
        edited_images: List[Union[str, Image.Image, torch.Tensor]],
    ) -> torch.Tensor:
        """
        Score edited images.

        Args:
            prompts: List of editing instruction prompts
            source_images: List of source images (path, PIL Image, or tensor)
            edited_images: List of edited images (path, PIL Image, or tensor)

        Returns:
            Tensor of scores with shape [batch_size, 1]
        """
        if self.rm_head_type == "ranknet_multi_head":
            batch_dim1 = self._build_batch(prompts, source_images, edited_images, reward_dim="dim1")
            batch_dim2 = self._build_batch(prompts, source_images, edited_images, reward_dim="dim2")
            rewards = self.model(return_dict=True, batch_dim1=batch_dim1, batch_dim2=batch_dim2)["logits"]
        else:
            batch = self._build_batch(prompts, source_images, edited_images, reward_dim=self.reward_dim)
            rewards = self.model(return_dict=True, **batch)["logits"]

        return rewards

    def __call__(
        self,
        prompts: List[str],
        source_images: List,
        edited_images: List,
    ) -> List[float]:
        """Callable interface for compatibility with reward function API."""
        rewards = self.score(prompts, source_images, edited_images)
        return rewards.squeeze(-1).tolist()


# ============================================================================
# Factory Function for rewards.py integration
# ============================================================================


def editreward_score(device: str, checkpoint_path: str, **kwargs):
    """
    Factory function for creating EditReward scorer.

    Usage in rewards.py:
        reward_fn:
          editreward:
            weight: 1.0
            checkpoint_path: /path/to/checkpoint
    """
    scorer = EditRewardScorer(checkpoint_path=checkpoint_path, device=device, **kwargs)

    def _score_fn(images, prompts, metadata):
        # images: edited images [B, C, H, W]
        # prompts: text prompts
        # metadata: should contain source_images
        source_images = metadata.get("source_images", metadata.get("input_images", []))

        if len(source_images) == 0:
            raise ValueError("source_images must be provided in metadata for EditReward")

        # Convert tensor images to PIL if needed
        edited_pil = []
        for img in images:
            if isinstance(img, torch.Tensor):
                if img.dim() == 3:
                    img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    edited_pil.append(Image.fromarray(img_np))
                else:
                    edited_pil.append(img)
            else:
                edited_pil.append(img)

        scores = scorer(prompts, source_images, edited_pil)
        return scores, {}

    return _score_fn


if __name__ == "__main__":
    """
    Example usage. Set environment variables before running:

        export EDITREWARD_MODEL_PATH=/path/to/MiMo-VL-7B-SFT-2508
        export EDITREWARD_CHECKPOINT_PATH=/path/to/EditReward-MiMo-VL-7B-SFT-2508
        export EDITREWARD_TEST_IMAGES=/path/to/test/images  # optional
        python editreward_scorer.py
    """
    import argparse

    parser = argparse.ArgumentParser(description="Test EditReward scorer")
    parser.add_argument(
        "--model-path",
        type=str,
        default=os.environ.get("EDITREWARD_MODEL_PATH"),
        help="Path to base model (or set EDITREWARD_MODEL_PATH env)",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=os.environ.get("EDITREWARD_CHECKPOINT_PATH"),
        help="Path to checkpoint (or set EDITREWARD_CHECKPOINT_PATH env)",
    )
    parser.add_argument(
        "--test-images",
        type=str,
        default=os.environ.get("EDITREWARD_TEST_IMAGES"),
        help="Path to test images directory (or set EDITREWARD_TEST_IMAGES env)",
    )
    args = parser.parse_args()

    if not args.model_path or not args.checkpoint_path:
        print("Error: Please set EDITREWARD_MODEL_PATH and EDITREWARD_CHECKPOINT_PATH environment variables")
        print("       or use --model-path and --checkpoint-path arguments")
        exit(1)

    scorer = EditRewardScorer(
        checkpoint_path=args.checkpoint_path,
        model_name_or_path=args.model_path,
        device="cuda",
        reward_dim="overall_detail",
        rm_head_type="ranknet_multi_head",
        pooling_strategy="mean",
        use_special_tokens=True,
        output_dim=2,
        rm_head_kwargs=None,
    )

    # Test with images if provided
    if args.test_images:
        source_path = os.path.join(args.test_images, "source_img_1.png")
        good_path = os.path.join(args.test_images, "target_img_2.png")
        bad_path = os.path.join(args.test_images, "target_img_1.png")
        prompt = "Add a green bowl on the branch"

        print("=" * 50)
        print("Test: Batch (good vs bad)")
        print("=" * 50)
        scores = scorer.score([prompt, prompt], [source_path, source_path], [good_path, bad_path])
        print(f"Good edit: {scores[0].tolist()}")
        print(f"Bad edit: {scores[1].tolist()}")
    else:
        print("Model loaded successfully!")
        print("Set EDITREWARD_TEST_IMAGES to run tests with sample images")
