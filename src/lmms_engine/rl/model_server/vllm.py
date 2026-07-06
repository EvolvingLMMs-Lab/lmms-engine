from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from lmms_engine.rl.lmms_eval.paths import ensure_lmms_eval_importable

ensure_lmms_eval_importable()

from lmms_eval.agentic.model_server.base import ModelServer  # noqa: E402
from lmms_eval.agentic.types import AgentInput, AgentOutput, ContentBlock  # noqa: E402

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
            raise ImportError(
                "VLLMChatModelServer requires `vllm`. Install vLLM in the rollout environment."
            ) from exc

        self.llm = LLM(model=model, **engine_kwargs)
        self.generation_kwargs = dict(generation_kwargs or {})
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.chat_template_content_format = chat_template_content_format
        self.mm_processor_kwargs = dict(mm_processor_kwargs or {})
        self.system_prompt = system_prompt
        self.default_max_tokens = int(default_max_tokens)
        self.weight_version_id: int | None = None
        self.weight_checkpoint_path: str | None = str(model)

    def generate(self, request: Any) -> AgentOutput:
        return self.generate_batch([request])[0]

    def generate_batch(self, requests: list[Any]) -> list[AgentOutput]:
        if not requests:
            return []
        for request in requests:
            if not isinstance(request, AgentInput):
                raise TypeError(
                    f"VLLMChatModelServer requires AgentInput requests, got {type(request).__name__}"
                )

        from vllm import SamplingParams

        messages = [self._request_to_messages(request) for request in requests]
        sampling_params = [
            SamplingParams(**self._sampling_kwargs(request)) for request in requests
        ]
        outputs = self.llm.chat(
            messages=messages,
            sampling_params=sampling_params,
            use_tqdm=False,
            chat_template_content_format=self.chat_template_content_format,
            chat_template_kwargs=self.chat_template_kwargs or None,
            mm_processor_kwargs=self.mm_processor_kwargs or None,
        )
        return [self._output_to_agent_output(output) for output in outputs]

    def update_weights(
        self,
        checkpoint_path: str | None = None,
        weights_path: str | None = None,
        version_id: int | None = None,
        metadata: dict[str, Any] | None = None,
        reload_kwargs: dict[str, Any] | None = None,
        is_checkpoint_format: bool = True,
        reset_prefix_cache: bool = True,
        sleep_level: int | None = None,
        sleep_mode: str = "wait",
        timeout_s: float | None = None,
        require_hf_checkpoint: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Reload vLLM weights in-place from a merged HF checkpoint path."""

        resolved_weights_path = weights_path or checkpoint_path
        if not resolved_weights_path:
            raise ValueError("VLLMChatModelServer.update_weights requires checkpoint_path or weights_path.")
        validation = self.validate_weight_path(
            resolved_weights_path,
            require_hf_checkpoint=require_hf_checkpoint,
        )
        resolved = Path(validation["resolved_path"])

        started = time.perf_counter()
        reload_options = {
            "weights_path": str(resolved),
            "is_checkpoint_format": bool(is_checkpoint_format),
            **dict(reload_kwargs or {}),
            **kwargs,
        }

        if sleep_level is not None:
            self.llm.sleep(level=int(sleep_level), mode=sleep_mode)
            if int(sleep_level) == 2:
                self.llm.wake_up(tags=["weights"])

        try:
            self._reload_weights(reload_options, timeout_s=timeout_s)
            if reset_prefix_cache and hasattr(self.llm, "reset_prefix_cache"):
                self.llm.reset_prefix_cache()
        finally:
            if sleep_level is not None:
                if int(sleep_level) == 2:
                    self.llm.wake_up(tags=["kv_cache"])
                else:
                    self.llm.wake_up()

        self.weight_version_id = version_id
        self.weight_checkpoint_path = str(resolved)
        return {
            "backend": "vllm",
            "version_id": version_id,
            "weights_path": str(resolved),
            "metadata": dict(metadata or {}),
            "validation": validation,
            "elapsed_s": time.perf_counter() - started,
        }

    def validate_weight_path(
        self,
        checkpoint_path: str,
        require_hf_checkpoint: bool = True,
    ) -> dict[str, Any]:
        return _validate_weight_path(checkpoint_path, require_hf_checkpoint=require_hf_checkpoint)

    def _reload_weights(self, reload_options: dict[str, Any], timeout_s: float | None = None) -> None:
        try:
            self.llm.collective_rpc("reload_weights", timeout=timeout_s, kwargs=reload_options)
        except TypeError as exc:
            if "is_checkpoint_format" not in str(exc):
                raise
            fallback_options = dict(reload_options)
            fallback_options.pop("is_checkpoint_format", None)
            self.llm.collective_rpc("reload_weights", timeout=timeout_s, kwargs=fallback_options)

    def _request_to_messages(self, request: AgentInput) -> list[dict[str, Any]]:
        if isinstance(request.metadata.get("messages"), list):
            return request.metadata["messages"]

        messages = []
        if self.system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.system_prompt}],
                }
            )
        history = request.metadata.get("conversation_history")
        if isinstance(history, list):
            messages.extend(
                message
                for message in (self._history_turn_to_message(turn) for turn in history)
                if message is not None
            )
        messages.append(
            {
                "role": request.metadata.get("role", "user"),
                "content": self._content_blocks_to_vllm_content(request.content),
            }
        )
        return messages

    def _content_blocks_to_vllm_content(
        self, blocks: list[ContentBlock]
    ) -> list[dict[str, Any]]:
        content = []
        for block in blocks:
            if block.type == "text" and block.data is not None:
                content.append({"type": "text", "text": str(block.data)})
            elif block.type in {"image", "image_url"} and block.data is not None:
                content.append(_image_content(block.data))
            elif block.type in {"video", "video_url"} and block.data is not None:
                content.append(_video_content(block.data))
            elif block.type in {"audio", "audio_url"} and block.data is not None:
                content.append(
                    {
                        "type": "audio_url",
                        "audio_url": {"url": _media_payload(block.data, "audio_url")},
                    }
                )
        if not content:
            content.append({"type": "text", "text": ""})
        return content

    def _history_turn_to_message(self, turn: Any) -> dict[str, Any] | None:
        if not isinstance(turn, dict):
            return None

        role = str(turn.get("role", "user"))
        content = turn.get("content", "")
        if role == "assistant":
            return {"role": role, "content": _history_text(content)}
        if isinstance(content, AgentInput):
            return {
                "role": role,
                "content": self._content_blocks_to_vllm_content(content.content),
            }
        if _is_content_block_list(content):
            return {
                "role": role,
                "content": self._content_blocks_to_vllm_content(content),
            }
        if isinstance(content, str):
            return {"role": role, "content": [{"type": "text", "text": content}]}
        if isinstance(content, list):
            return {"role": role, "content": content}
        return {"role": role, "content": [{"type": "text", "text": str(content)}]}

    def _sampling_kwargs(self, request: AgentInput) -> dict[str, Any]:
        generation_kwargs = dict(self.generation_kwargs)
        generation_kwargs.update(request.generation_kwargs or {})
        max_tokens = int(
            generation_kwargs.pop(
                "max_tokens",
                generation_kwargs.pop("max_new_tokens", self.default_max_tokens),
            )
        )
        kwargs = {
            "max_tokens": max_tokens,
            "temperature": generation_kwargs.pop("temperature", 0),
            "top_p": generation_kwargs.pop("top_p", 1.0),
        }
        stop = generation_kwargs.pop("stop", None) or generation_kwargs.pop(
            "until", None
        )
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

    return {
        "type": "video_url",
        "video_url": {"url": encode_video_url(_frames_to_numpy(payload))},
    }


def _frames_to_numpy(frames: Any):
    import numpy as np

    if hasattr(frames, "ndim"):
        array = np.asarray(frames)
        if array.ndim == 3:
            array = array[None, ...]
        return array
    if not isinstance(frames, list | tuple):
        frames = [frames]
    return np.stack(
        [
            np.asarray(frame.convert("RGB") if hasattr(frame, "convert") else frame)
            for frame in frames
        ],
        axis=0,
    )


def _is_content_block_list(value: Any) -> bool:
    return isinstance(value, list) and all(
        isinstance(item, ContentBlock) for item in value
    )


def _history_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, AgentOutput):
        text = content.first_text()
        return "" if text is None else str(text)
    if isinstance(content, list):
        parts = []
        for item in content:
            if (
                isinstance(item, ContentBlock)
                and item.type == "text"
                and item.data is not None
            ):
                parts.append(str(item.data))
            elif (
                isinstance(item, dict)
                and item.get("type") == "text"
                and item.get("text") is not None
            ):
                parts.append(str(item["text"]))
        return "\n".join(parts)
    return "" if content is None else str(content)


def _validate_weight_path(checkpoint_path: str, require_hf_checkpoint: bool = True) -> dict[str, Any]:
    resolved = Path(checkpoint_path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"vLLM weight checkpoint does not exist: {resolved}")
    if not resolved.is_dir():
        raise FileNotFoundError(f"vLLM weight checkpoint must be a directory: {resolved}")

    config_path = resolved / "config.json"
    safetensors_files = sorted(resolved.glob("*.safetensors"))
    bin_files = sorted(resolved.glob("*.bin"))
    index_files = sorted(resolved.glob("*.index.json"))
    if require_hf_checkpoint and not config_path.exists():
        raise FileNotFoundError(f"vLLM HF checkpoint is missing config.json: {resolved}")
    if require_hf_checkpoint and not (safetensors_files or bin_files or index_files):
        raise FileNotFoundError(
            "vLLM HF checkpoint has no model weight files (*.safetensors, *.bin, or *.index.json): "
            f"{resolved}"
        )

    return {
        "resolved_path": str(resolved),
        "require_hf_checkpoint": bool(require_hf_checkpoint),
        "has_config": config_path.exists(),
        "num_safetensors": len(safetensors_files),
        "num_bin": len(bin_files),
        "num_index": len(index_files),
    }
