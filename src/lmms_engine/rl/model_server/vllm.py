from __future__ import annotations

from typing import Any

from lmms_engine.rl.lmms_eval.paths import ensure_lmms_eval_importable

ensure_lmms_eval_importable()

from lmms_eval.agentic.model_server.base import ModelServer
from lmms_eval.agentic.types import AgentInput, AgentOutput, ContentBlock

_AGENTIC_ONLY_KEYS = {"max_agentic_steps", "max_game_steps", "game_seed"}
_GENERATION_KEYS_TO_DROP = {"do_sample", "num_beams"}


class VLLMChatModelServer(ModelServer):
    """In-process vLLM chat backend for Ray actor model serving.

    This is intended to run inside a Ray model-server actor. It avoids the
    OpenAI HTTP/base64 path and passes Python multimodal objects through Ray.
    """

    def __init__(
        self,
        model: str,
        generation_kwargs: dict[str, Any] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        chat_template_content_format: str = "openai",
        mm_processor_kwargs: dict[str, Any] | None = None,
        system_prompt: str | None = None,
        default_max_tokens: int = 64,
        **engine_kwargs: Any,
    ) -> None:
        try:
            from vllm import LLM
        except ImportError as exc:
            raise ImportError("VLLMChatModelServer requires `vllm`. Install vLLM in the rollout environment.") from exc

        self.llm = LLM(model=model, **engine_kwargs)
        self.generation_kwargs = dict(generation_kwargs or {})
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.chat_template_content_format = chat_template_content_format
        self.mm_processor_kwargs = dict(mm_processor_kwargs or {})
        self.system_prompt = system_prompt
        self.default_max_tokens = int(default_max_tokens)

    def generate(self, request: Any) -> AgentOutput:
        return self.generate_batch([request])[0]

    def generate_batch(self, requests: list[Any]) -> list[AgentOutput]:
        if not requests:
            return []
        for request in requests:
            if not isinstance(request, AgentInput):
                raise TypeError(f"VLLMChatModelServer requires AgentInput requests, got {type(request).__name__}")

        from vllm import SamplingParams

        messages = [self._request_to_messages(request) for request in requests]
        sampling_params = [SamplingParams(**self._sampling_kwargs(request)) for request in requests]
        outputs = self.llm.chat(
            messages=messages,
            sampling_params=sampling_params,
            use_tqdm=False,
            chat_template_content_format=self.chat_template_content_format,
            chat_template_kwargs=self.chat_template_kwargs or None,
            mm_processor_kwargs=self.mm_processor_kwargs or None,
        )
        return [self._output_to_agent_output(output) for output in outputs]

    def _request_to_messages(self, request: AgentInput) -> list[dict[str, Any]]:
        if isinstance(request.metadata.get("messages"), list):
            return request.metadata["messages"]
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": self.system_prompt}]})
        messages.append(
            {
                "role": request.metadata.get("role", "user"),
                "content": self._content_blocks_to_vllm_content(request.content),
            }
        )
        return messages

    def _content_blocks_to_vllm_content(self, blocks: list[ContentBlock]) -> list[dict[str, Any]]:
        content = []
        for block in blocks:
            if block.type == "text" and block.data is not None:
                content.append({"type": "text", "text": str(block.data)})
            elif block.type in {"image", "image_url"} and block.data is not None:
                content.append(_image_content(block.data))
            elif block.type in {"video", "video_url"} and block.data is not None:
                content.append(_video_content(block.data))
            elif block.type in {"audio", "audio_url"} and block.data is not None:
                content.append({"type": "audio_url", "audio_url": {"url": _media_payload(block.data, "audio_url")}})
        if not content:
            content.append({"type": "text", "text": ""})
        return content

    def _sampling_kwargs(self, request: AgentInput) -> dict[str, Any]:
        generation_kwargs = dict(self.generation_kwargs)
        generation_kwargs.update(request.generation_kwargs or {})
        max_tokens = int(
            generation_kwargs.pop("max_tokens", generation_kwargs.pop("max_new_tokens", self.default_max_tokens))
        )
        kwargs = {
            "max_tokens": max_tokens,
            "temperature": generation_kwargs.pop("temperature", 0),
            "top_p": generation_kwargs.pop("top_p", 1.0),
        }
        stop = generation_kwargs.pop("stop", None) or generation_kwargs.pop("until", None)
        if stop is not None:
            kwargs["stop"] = stop if isinstance(stop, list) else [stop]
        for key in _AGENTIC_ONLY_KEYS | _GENERATION_KEYS_TO_DROP:
            generation_kwargs.pop(key, None)
        kwargs.update(generation_kwargs)
        return kwargs

    @staticmethod
    def _output_to_agent_output(output: Any) -> AgentOutput:
        response = output.outputs[0] if getattr(output, "outputs", None) else None
        text = getattr(response, "text", "") if response is not None else ""
        metadata = {
            "raw_response": output,
            "token_ids": getattr(response, "token_ids", None),
            "logprobs": getattr(response, "logprobs", None),
            "finish_reason": getattr(response, "finish_reason", None),
        }
        return AgentOutput(content=[ContentBlock.text(text or "")], metadata=metadata)


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


def _image_content(data: Any) -> dict[str, Any]:
    payload = _media_payload(data, "image_url")
    if isinstance(payload, str):
        return {"type": "image_url", "image_url": {"url": payload}}
    return {"type": "image_pil", "image_pil": payload}


def _video_content(data: Any) -> dict[str, Any]:
    payload = _media_payload(data, "video_url")
    if isinstance(payload, str):
        return {"type": "video_url", "video_url": {"url": payload}}

    from vllm.multimodal.utils import encode_video_url

    return {"type": "video_url", "video_url": {"url": encode_video_url(_frames_to_numpy(payload))}}


def _frames_to_numpy(frames: Any):
    import numpy as np

    if hasattr(frames, "ndim"):
        array = np.asarray(frames)
        if array.ndim == 3:
            array = array[None, ...]
        return array
    if not isinstance(frames, list | tuple):
        frames = [frames]
    return np.stack([np.asarray(frame.convert("RGB") if hasattr(frame, "convert") else frame) for frame in frames], axis=0)
