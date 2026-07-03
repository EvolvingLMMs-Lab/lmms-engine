#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-examples/rl/vizdoom_qwen3_5_4b.yaml}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
PYTHON="${PYTHON:-.venv/bin/python}"
SMOKE="${SMOKE:-0}"
MODEL_SERVER="${MODEL_SERVER:-}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

export CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS
export NCCL_ASYNC_ERROR_HANDLING
export TORCH_NCCL_ASYNC_ERROR_HANDLING

TMP_CONFIG=""
cleanup() {
  if [[ -n "${TMP_CONFIG}" && -f "${TMP_CONFIG}" ]]; then
    rm -f "${TMP_CONFIG}"
  fi
}
trap cleanup EXIT

TMP_CONFIG="$(mktemp "${TMPDIR:-/tmp}/lmms-engine-vizdoom-rl.XXXXXX.yaml")"

CONFIG_IN="${CONFIG}" CONFIG_OUT="${TMP_CONFIG}" SMOKE="${SMOKE}" MODEL_SERVER="${MODEL_SERVER}" "${PYTHON}" - <<'PY'
import os
from pathlib import Path

import yaml


def _env(name, default=None):
    value = os.getenv(name)
    return default if value is None or value == "" else value


def _as_bool(value):
    return str(value).lower() in {"1", "true", "yes", "on"}


def _as_int(name, default=None):
    value = _env(name)
    if value is None:
        return default
    return int(value)


def _set_if_env(mapping, key, env_name, cast=lambda value: value):
    value = _env(env_name)
    if value is not None:
        mapping[key] = cast(value)


config_path = Path(os.environ["CONFIG_IN"])
with config_path.open("r") as f:
    config = yaml.safe_load(f)

trainer_args = config.setdefault("trainer_args", {})
rl_config = trainer_args.setdefault("rl_config", {})
task = rl_config.setdefault("task", {})
rollout = rl_config.setdefault("rollout", {})
data_buffer = rl_config.setdefault("data_buffer", {})
training = rl_config.setdefault("training", {})
dataset_config = config.setdefault("dataset_config", {})
processor_config = dataset_config.setdefault("processor_config", {})
model_config = config.setdefault("model_config", {})

model_name = _env("MODEL_NAME") or _env("TRAIN_MODEL")
if model_name:
    model_config["load_from_pretrained_path"] = model_name
    processor_config["processor_name"] = model_name
    if "Qwen3-VL" in model_name and _env("FSDP_LAYER_CLS") is None:
        trainer_args.setdefault("fsdp_config", {})["transformer_layer_cls_to_wrap"] = ["Qwen3VLTextDecoderLayer"]

fsdp_layer_cls = _env("FSDP_LAYER_CLS")
if fsdp_layer_cls:
    trainer_args.setdefault("fsdp_config", {})["transformer_layer_cls_to_wrap"] = [
        item.strip() for item in fsdp_layer_cls.split(",") if item.strip()
    ]

model_server = (_env("MODEL_SERVER") or "").lower()
if model_server == "debug":
    rl_config["model_server"] = {"name": "debug", "action": _env("DEBUG_ACTION", "ATTACK")}
elif model_server in {"ray", "ray_debug"}:
    if model_server == "ray_debug" or _env("RAY_MODEL_SERVER_FACTORY") is None:
        server_spec = {"name": "debug", "action": _env("DEBUG_ACTION", "ATTACK")}
    else:
        server_spec = {
            "factory": _env("RAY_MODEL_SERVER_FACTORY"),
            "model": _env("POLICY_MODEL", _env("MODEL_NAME", "Qwen/Qwen3.5-4B")),
        }
    rl_config["model_server"] = {
        "name": "ray_actor_pool",
        "actor_name_prefix": _env("RAY_ACTOR_NAME_PREFIX", "qwen3-5-4b-vizdoom-policy"),
        "num_replicas": _as_int("RAY_MODEL_SERVER_REPLICAS", 1),
        "actor_options": {},
        "server": server_spec,
    }
elif model_server == "openai":
    server = dict(rl_config.get("model_server") or {})
    server["name"] = "openai"
    _set_if_env(server, "model", "POLICY_MODEL")
    _set_if_env(server, "model", "OPENAI_MODEL")
    _set_if_env(server, "base_url", "OPENAI_BASE_URL")
    _set_if_env(server, "api_key", "OPENAI_API_KEY")
    rl_config["model_server"] = server

policy_model = _env("POLICY_MODEL") or _env("OPENAI_MODEL")
if policy_model and isinstance(rl_config.get("model_server"), dict):
    rl_config["model_server"]["model"] = policy_model

_set_if_env(trainer_args, "output_dir", "OUTPUT_DIR")
_set_if_env(trainer_args, "max_steps", "MAX_STEPS", int)
_set_if_env(trainer_args, "save_steps", "SAVE_STEPS", int)
_set_if_env(task, "limit", "TASK_LIMIT", int)
_set_if_env(task, "repeats", "TASK_REPEATS", int)
_set_if_env(task, "max_steps", "TASK_MAX_STEPS", int)
_set_if_env(rollout, "num_workers", "ROLLOUT_WORKERS", int)
_set_if_env(data_buffer, "train_batch_size", "TRAIN_BATCH_SIZE", int)
_set_if_env(data_buffer, "min_trajectories_per_batch", "MIN_TRAJECTORIES_PER_BATCH", int)
_set_if_env(training, "global_batch_size", "RL_GLOBAL_BATCH_SIZE", int)

save_final_checkpoint = _env("SAVE_FINAL_CHECKPOINT")
if save_final_checkpoint is not None:
    rl_config["save_final_checkpoint"] = _as_bool(save_final_checkpoint)

if _as_bool(os.environ.get("SMOKE", "0")):
    trainer_args["output_dir"] = _env("OUTPUT_DIR", "./output/qwen3_5_4b_vizdoom_rl_smoke")
    trainer_args["max_steps"] = _as_int("MAX_STEPS", 1)
    trainer_args["save_steps"] = _as_int("SAVE_STEPS", 1000000)
    task["limit"] = _as_int("TASK_LIMIT", 1)
    task["repeats"] = _as_int("TASK_REPEATS", 1)
    task["max_steps"] = _as_int("TASK_MAX_STEPS", 1)
    rollout["num_workers"] = _as_int("ROLLOUT_WORKERS", 1)
    data_buffer["max_trajectories"] = _as_int("MAX_TRAJECTORIES", 4)
    data_buffer["high_watermark"] = _as_int("HIGH_WATERMARK", 3)
    data_buffer["low_watermark"] = _as_int("LOW_WATERMARK", 1)
    data_buffer["train_batch_size"] = _as_int("TRAIN_BATCH_SIZE", 1)
    data_buffer["min_trajectories_per_batch"] = _as_int("MIN_TRAJECTORIES_PER_BATCH", 1)
    training["global_batch_size"] = _as_int("RL_GLOBAL_BATCH_SIZE", 1)
    rl_config["save_final_checkpoint"] = _as_bool(_env("SAVE_FINAL_CHECKPOINT", "0"))

with Path(os.environ["CONFIG_OUT"]).open("w") as f:
    yaml.safe_dump(config, f, sort_keys=False)
PY

echo "Using generated config: ${TMP_CONFIG}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} NPROC_PER_NODE=${NPROC_PER_NODE} SMOKE=${SMOKE}"

"${PYTHON}" -m torch.distributed.run --standalone --nproc-per-node="${NPROC_PER_NODE}" \
  -m lmms_engine.launch.cli \
  config_yaml="${TMP_CONFIG}"
