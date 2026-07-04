from __future__ import annotations

import json
from typing import Any

import torch
from loguru import logger

from lmms_engine.rl.lmms_eval.paths import ensure_lmms_eval_importable

ensure_lmms_eval_importable()

from lmms_eval.agentic.model_server.base import ModelServer
from lmms_eval.agentic.types import AgentInput, AgentOutput, ContentBlock

_AGENTIC_ONLY_KEYS = {"max_agentic_steps", "max_game_steps", "game_seed"}
_STOP_KEYS = {"stop", "stop_strings", "until"}


class TransformersChatModelServer(ModelServer):
    """In-process Transformers chat backend for Ray actor model serving."""

    def __init__(
        self,
        model: str,
        generation_kwargs: dict[str, Any] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        processor_kwargs: dict[str, Any] | None = None,
        model_kwargs: dict[str, Any] | None = None,
        system_prompt: str | None = None,
        torch_dtype: str | torch.dtype | None = "bfloat16",
        device: str = "cuda",
        device_map: str | None = "auto",
        attn_implementation: str | None = "sdpa",
        trust_remote_code: bool = True,
        local_files_only: bool = False,
        default_max_new_tokens: int = 64,
    ) -> None:
        from transformers import AutoProcessor, AutoTokenizer

        model_kwargs = dict(model_kwargs or {})
        if attn_implementation is not None:
            model_kwargs.setdefault("attn_implementation", attn_implementation)
        if device_map is not None:
            model_kwargs.setdefault("device_map", device_map)
        dtype_value = _resolve_dtype(torch_dtype)
        if dtype_value is not None:
            model_kwargs.setdefault(_dtype_kwarg(model), dtype_value)
        model_kwargs.setdefault("trust_remote_code", trust_remote_code)
        model_kwargs.setdefault("local_files_only", local_files_only)

        model_cls = _resolve_model_class(model)
        self.model = model_cls.from_pretrained(model, **model_kwargs).eval()
        if device_map is None:
            self.model.to(device)

        processor_kwargs = dict(processor_kwargs or {})
        processor_kwargs.setdefault("trust_remote_code", trust_remote_code)
        processor_kwargs.setdefault("local_files_only", local_files_only)
        self.processor = AutoProcessor.from_pretrained(model, **processor_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        self.generation_kwargs = dict(generation_kwargs or {})
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.system_prompt = system_prompt
        self.device = device
        self.device_map = device_map
        self.default_max_new_tokens = int(default_max_new_tokens)
        logger.info(
            "Loaded TransformersChatModelServer "
            f"model={model}, model_cls={model_cls.__name__}, device_map={device_map}, device={device}"
        )

    def generate(self, request: Any) -> AgentOutput:
        if not isinstance(request, AgentInput):
            raise TypeError(f"TransformersChatModelServer requires AgentInput requests, got {type(request).__name__}")
        messages = self._request_to_messages(request)
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **self.chat_template_kwargs,
        )
        inputs = self._processor_inputs(text, messages)
        generate_kwargs = self._generation_kwargs(request)

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **generate_kwargs)

        input_len = inputs["input_ids"].shape[-1]
        generated_ids = output_ids[:, input_len:]
        text_outputs = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        response_text = _truncate_at_stop(text_outputs[0] if text_outputs else "", request.generation_kwargs)
        return AgentOutput(
            content=[ContentBlock.text(response_text)],
            metadata={
                "model_server": "transformers",
                "output_tokens": int(generated_ids.shape[-1]) if generated_ids.ndim == 2 else None,
            },
        )

    def generate_batch(self, requests: list[Any]) -> list[AgentOutput]:
        return [self.generate(request) for request in requests]

    def _request_to_messages(self, request: AgentInput) -> list[dict[str, Any]]:
        if isinstance(request.metadata.get("messages"), list):
            return request.metadata["messages"]

        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": self.system_prompt}]})
        messages.append(
            {
                "role": request.metadata.get("role", "user"),
                "content": self._content_blocks_to_qwen_content(request.content),
            }
        )
        return messages

    def _content_blocks_to_qwen_content(self, blocks: list[ContentBlock]) -> list[dict[str, Any]]:
        content = []
        for block in blocks:
            if block.type == "text" and block.data is not None:
                content.append({"type": "text", "text": str(block.data)})
            elif block.type in {"image", "image_url"} and block.data is not None:
                content.append({"type": "image", "image": _media_payload(block.data, "image_url")})
            elif block.type in {"video", "video_url"} and block.data is not None:
                content.append({"type": "video", "video": _media_payload(block.data, "video_url")})
            elif block.type == "vizdoom_state" and block.data is not None:
                content.append({"type": "text", "text": _state_text(block.data)})
        if not content:
            content.append({"type": "text", "text": ""})
        return content

    def _processor_inputs(self, text: str, messages: list[dict[str, Any]]) -> Any:
        try:
            from qwen_vl_utils import process_vision_info

            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages,
                return_video_kwargs=True,
                image_patch_size=16,
                return_video_metadata=True,
            )
            video_metadata = None
            if video_inputs is not None:
                video_inputs, video_metadata = zip(*video_inputs)
                video_inputs, video_metadata = list(video_inputs), list(video_metadata)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                video_metadata=video_metadata,
                **video_kwargs,
                return_tensors="pt",
            )
        except ImportError:
            inputs = self.processor(text=[text], return_tensors="pt")
        return inputs.to(_input_device(self.device, self.device_map))

    def _generation_kwargs(self, request: AgentInput) -> dict[str, Any]:
        kwargs = dict(self.generation_kwargs)
        kwargs.update(request.generation_kwargs or {})
        kwargs["max_new_tokens"] = int(
            kwargs.pop("max_tokens", kwargs.get("max_new_tokens", self.default_max_new_tokens))
        )
        for key in _AGENTIC_ONLY_KEYS | _STOP_KEYS:
            kwargs.pop(key, None)
        if kwargs.get("temperature", 0) and "do_sample" not in kwargs:
            kwargs["do_sample"] = True
        if not kwargs.get("do_sample", False):
            kwargs.pop("temperature", None)
            kwargs.pop("top_p", None)
            kwargs.pop("top_k", None)
        kwargs.setdefault("eos_token_id", self.tokenizer.eos_token_id)
        if self.tokenizer.pad_token_id is not None:
            kwargs.setdefault("pad_token_id", self.tokenizer.pad_token_id)
        return kwargs


def _resolve_model_class(model: str) -> Any:
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
    )

    config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    model_type = getattr(config, "model_type", "")
    if "qwen3_5" in model_type:
        from transformers import (
            Qwen3_5ForConditionalGeneration,
            Qwen3_5MoeForConditionalGeneration,
        )

        return Qwen3_5MoeForConditionalGeneration if "moe" in model_type else Qwen3_5ForConditionalGeneration
    if "qwen3_vl" in model_type:
        from transformers import (
            Qwen3VLForConditionalGeneration,
            Qwen3VLMoeForConditionalGeneration,
        )

        return Qwen3VLMoeForConditionalGeneration if "moe" in model_type else Qwen3VLForConditionalGeneration
    if type(config) in AutoModelForImageTextToText._model_mapping.keys():
        return AutoModelForImageTextToText
    return AutoModelForCausalLM


def _dtype_kwarg(model: str) -> str:
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    model_type = getattr(config, "model_type", "")
    return "torch_dtype" if "qwen3_5" in model_type else "dtype"


def _resolve_dtype(value: str | torch.dtype | None) -> str | torch.dtype | None:
    if value is None or isinstance(value, torch.dtype):
        return value
    if value == "auto":
        return "auto"
    if hasattr(torch, value):
        return getattr(torch, value)
    raise ValueError(f"Unknown torch dtype: {value}")


def _input_device(device: str, device_map: str | None) -> str:
    if device_map == "auto" and torch.cuda.is_available():
        return "cuda"
    return device


def _media_payload(data: Any, nested_key: str) -> Any:
    if isinstance(data, dict):
        if "url" in data:
            return data["url"]
        nested = data.get(nested_key)
        if isinstance(nested, dict) and "url" in nested:
            return nested["url"]
        if nested is not None:
            return nested
    return data


def _state_text(data: Any) -> str:
    try:
        return "Structured VizDoom state: " + json.dumps(data, ensure_ascii=False, sort_keys=True, default=str)
    except TypeError:
        return f"Structured VizDoom state: {data}"


def _truncate_at_stop(text: str, generation_kwargs: dict[str, Any] | None) -> str:
    generation_kwargs = generation_kwargs or {}
    stops = generation_kwargs.get("stop") or generation_kwargs.get("stop_strings") or generation_kwargs.get("until")
    if stops is None:
        return text
    if isinstance(stops, str):
        stops = [stops]
    for stop in stops:
        index = text.find(str(stop))
        if index >= 0:
            text = text[:index]
    return text
