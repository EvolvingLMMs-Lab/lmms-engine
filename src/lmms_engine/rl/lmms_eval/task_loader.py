from __future__ import annotations

import copy
import inspect
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import datasets

from lmms_engine.rl.lmms_eval.paths import ensure_lmms_eval_importable

ensure_lmms_eval_importable()

from lmms_eval.agentic.rollout import RolloutEpisodeSpec
from lmms_eval import utils as lmms_eval_utils


@dataclass(slots=True)
class LMMSEvalRolloutTaskConfig:
    """Resolve lmms-eval rollout components with lmms-engine training docs."""

    task_name: str | None = None
    task_yaml: str | None = None
    include_path: str | list[str] | None = None
    data_path: str | None = None
    data_format: str = "jsonl"
    docs: list[dict[str, Any]] | None = None
    split: str | None = None
    limit: int | None = None
    offset: int = 0
    repeats: int = 1
    model_server: Any = None
    model_output_parser: Any = None
    loop_worker: Any = "simple"
    max_steps: int | None = None
    seed: int | None = None
    generation_kwargs: dict[str, Any] = field(default_factory=dict)
    lmms_eval_specific_kwargs: dict[str, Any] = field(default_factory=dict)
    request_metadata: dict[str, Any] = field(default_factory=dict)


def build_rollout_episode_specs(config: LMMSEvalRolloutTaskConfig | dict[str, Any] | None = None) -> list[RolloutEpisodeSpec]:
    task_config = _coerce_task_config(config)
    task = _load_task_config(task_config)
    docs = _load_rollout_docs(task_config)
    lmms_eval_kwargs = _resolve_lmms_eval_kwargs(task.get("lmms_eval_specific_kwargs"), model_name=None)

    specs: list[RolloutEpisodeSpec] = []
    for doc_idx, doc in docs:
        for repeat_idx in range(max(1, task_config.repeats)):
            seed = None if task_config.seed is None else int(task_config.seed) + doc_idx * max(1, task_config.repeats) + repeat_idx
            specs.append(
                RolloutEpisodeSpec(
                    doc=doc,
                    game_env=_serializable_component_spec(task["game_env"]),
                    observation_parser=_serializable_component_spec(task["observation_parser"]),
                    action_parser=_serializable_component_spec(task["action_parser"]),
                    model_server=task_config.model_server or "openai",
                    loop_worker=_serializable_component_spec(task_config.loop_worker),
                    model_output_parser=_serializable_component_spec(task_config.model_output_parser),
                    generation_kwargs={**(task.get("generation_kwargs") or {}), **task_config.generation_kwargs},
                    lmms_eval_specific_kwargs={
                        **lmms_eval_kwargs,
                        **task_config.lmms_eval_specific_kwargs,
                    },
                    max_steps=int(task_config.max_steps or (task.get("generation_kwargs") or {}).get("max_game_steps", 32)),
                    seed=seed,
                    request_metadata={
                        "task": task["task"],
                        "doc_id": doc_idx,
                        "repeat_idx": repeat_idx,
                        **task_config.request_metadata,
                    },
                )
            )
    return specs


def _coerce_task_config(config: LMMSEvalRolloutTaskConfig | dict[str, Any] | None) -> LMMSEvalRolloutTaskConfig:
    if config is None:
        return LMMSEvalRolloutTaskConfig()
    if isinstance(config, LMMSEvalRolloutTaskConfig):
        return config
    allowed = set(LMMSEvalRolloutTaskConfig.__dataclass_fields__)
    return LMMSEvalRolloutTaskConfig(**{key: value for key, value in dict(config).items() if key in allowed})


def _load_task_config(config: LMMSEvalRolloutTaskConfig) -> dict[str, Any]:
    if config.task_yaml is not None:
        yaml_path = str(Path(config.task_yaml).expanduser())
    else:
        if config.task_name is None:
            raise ValueError("Set rl_config.task.task_name or rl_config.task.task_yaml for lmms-eval rollout.")
        yaml_path = str(_find_task_yaml(config.task_name, config.include_path))
    return lmms_eval_utils.load_yaml_config(yaml_path=yaml_path, mode="full")


def _load_rollout_docs(config: LMMSEvalRolloutTaskConfig) -> list[tuple[int, Any]]:
    if config.docs is not None:
        docs = list(config.docs)
    elif config.data_path is not None:
        docs = _load_docs_from_path(config.data_path, config.data_format)
    else:
        raise ValueError(
            "RL rollout docs must come from lmms-engine data. Set rl_config.task.data_path "
            "or rl_config.task.docs; lmms-eval task YAML is only used for env/loop/parser components."
        )

    start = max(0, int(config.offset))
    end = None if config.limit is None else start + max(0, int(config.limit))
    return [(idx, docs[idx]) for idx in range(start, len(docs) if end is None else min(end, len(docs)))]


def _load_docs_from_path(path: str, data_format: str) -> list[dict[str, Any]]:
    resolved = _resolve_engine_data_path(path)
    normalized_format = data_format.lower()
    if normalized_format == "jsonl":
        with open(resolved, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    if normalized_format == "json":
        with open(resolved, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            split = data.get("train") or data.get("data")
            if isinstance(split, list):
                return split
        if isinstance(data, list):
            return data
        raise ValueError(f"JSON rollout data must be a list or contain a train/data list: {resolved}")

    dataset = datasets.load_dataset(normalized_format, data_files=str(resolved), split="train")
    return [dict(item) for item in dataset]


def _resolve_engine_data_path(path: str) -> Path:
    expanded = Path(os.path.expandvars(os.path.expanduser(path)))
    if expanded.is_absolute():
        return expanded
    candidates = [
        Path.cwd() / expanded,
        _lmms_engine_package_parent() / expanded,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return expanded


def _find_task_yaml(task_name: str, include_path: str | list[str] | None) -> Path:
    roots = [_lmms_eval_package_parent() / "lmms_eval" / "tasks"]
    if include_path is not None:
        if isinstance(include_path, str):
            roots.append(Path(include_path).expanduser())
        else:
            roots.extend(Path(path).expanduser() for path in include_path)

    for root in roots:
        for yaml_path in root.rglob("*.yaml"):
            try:
                config = lmms_eval_utils.load_yaml_config(yaml_path=str(yaml_path), mode="simple")
            except Exception:
                continue
            if config.get("task") == task_name:
                return yaml_path
    raise ValueError(f"Could not find lmms-eval task YAML for task {task_name!r}.")


def _resolve_local_data_files(config: dict[str, Any], yaml_path: str) -> None:
    dataset_kwargs = config.get("dataset_kwargs")
    if not isinstance(dataset_kwargs, dict):
        return
    data_files = dataset_kwargs.get("data_files")
    if data_files is None:
        return

    yaml_dir = Path(yaml_path).resolve().parent
    package_root = _lmms_eval_package_parent()
    dataset_kwargs["data_files"] = _rewrite_data_files(data_files, yaml_dir, package_root)


def _rewrite_data_files(value: Any, yaml_dir: Path, package_root: Path) -> Any:
    if isinstance(value, str):
        return str(_resolve_data_file(value, yaml_dir, package_root))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_rewrite_data_files(item, yaml_dir, package_root) for item in value]
    if isinstance(value, dict):
        return {key: _rewrite_data_files(item, yaml_dir, package_root) for key, item in value.items()}
    return value


def _resolve_data_file(path: str, yaml_dir: Path, package_root: Path) -> Path:
    expanded = Path(os.path.expandvars(os.path.expanduser(path)))
    if expanded.is_absolute() and expanded.exists():
        return expanded
    candidates = [
        Path.cwd() / expanded,
        yaml_dir / expanded,
        package_root / expanded,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return expanded


def _lmms_eval_package_parent() -> Path:
    return ensure_lmms_eval_importable()


def _lmms_engine_package_parent() -> Path:
    import lmms_engine

    return Path(lmms_engine.__file__).resolve().parent.parent.parent


def _resolve_lmms_eval_kwargs(value: Any, model_name: str | None = None) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    if model_name and isinstance(value.get(model_name), dict):
        return dict(value[model_name])
    resolved = dict(value)
    if isinstance(value.get("default"), dict):
        resolved.update(value["default"])
    if isinstance(value.get("dataset"), dict):
        resolved.update(value["dataset"])
    return resolved


def _serializable_component_spec(value: Any) -> Any:
    if callable(value):
        source = inspect.getsourcefile(value)
        name = getattr(value, "__name__", None)
        if source and name:
            return f"{Path(source).resolve()}:{name}"
        module = getattr(value, "__module__", None)
        qualname = getattr(value, "__qualname__", None)
        if module and qualname:
            return f"{module}:{qualname}"
        return value
    if isinstance(value, dict):
        return {key: _serializable_component_spec(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serializable_component_spec(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_serializable_component_spec(item) for item in value)
    return value


def clone_rollout_spec(spec: RolloutEpisodeSpec, *, seed: int | None = None) -> RolloutEpisodeSpec:
    cloned = copy.copy(spec)
    if seed is not None:
        cloned.seed = seed
    return cloned
