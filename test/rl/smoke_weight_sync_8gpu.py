from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT / "src", REPO_ROOT / "src" / "lmms-eval"):
    sys.path.insert(0, str(path))

import ray  # noqa: E402

from lmms_engine.rl.protocol import ModelVersion  # noqa: E402
from lmms_engine.rl.training_engine.disk_delta import publish_delta_checkpoint  # noqa: E402
from lmms_engine.rl.training_engine.weight_sync import (  # noqa: E402
    RayActorWeightSyncClient,
)


def main() -> None:
    args = _parse_args()

    sync_root = Path(args.sync_dir).expanduser().resolve() if args.sync_dir else _default_sync_root()
    if sync_root.exists() and args.clean:
        shutil.rmtree(sync_root)
    if args.mode == "full":
        checkpoint_path, metadata, version_id = _prepare_full_smoke_checkpoint(sync_root)
    else:
        checkpoint_path, metadata, version_id = _prepare_delta_smoke_checkpoint(sync_root)

    ray.shutdown()
    ray.init(
        num_gpus=args.num_gpus,
        include_dashboard=False,
        ignore_reinit_error=True,
        runtime_env={
            "env_vars": {
                "PYTHONPATH": os.pathsep.join([str(REPO_ROOT / "src"), str(REPO_ROOT / "src" / "lmms-eval")]),
            }
        },
    )

    try:
        actors = [WeightSyncSmokeActor.remote(index=i) for i in range(args.num_gpus)]
        statuses = ray.get([actor.status.remote() for actor in actors], timeout=args.timeout_s)
        _assert_eight_gpu_placement(statuses, args.num_gpus)

        client = RayActorWeightSyncClient(
            actor_handles=actors,
            timeout_s=args.timeout_s,
            preflight_weight_path=True,
            require_hf_checkpoint=True,
            extra_kwargs={"reset_prefix_cache": False, "timeout_s": args.timeout_s},
        )
        result = client.reload_weights(
            ModelVersion(
                version_id=version_id,
                checkpoint_path=str(checkpoint_path),
                metadata=metadata,
            )
        )
        _assert_update_result(result, args.num_gpus)
        print(
            json.dumps(
                {
                    "status": "passed",
                    "mode": args.mode,
                    "num_gpus": args.num_gpus,
                    "checkpoint_path": str(checkpoint_path),
                    "actor_statuses": statuses,
                    "weight_sync": result,
                },
                indent=2,
                sort_keys=True,
            )
        )
    finally:
        ray.shutdown()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an 8-GPU Ray weight-sync smoke test.")
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--sync-dir", type=str, default=None)
    parser.add_argument("--mode", choices=["full", "delta"], default="full")
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--clean", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _default_sync_root() -> Path:
    base = REPO_ROOT / "output" / "weight_sync_smoke"
    base.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="run-", dir=base))


def _write_hf_checkpoint_stub(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"stub")


def _prepare_full_smoke_checkpoint(sync_root: Path) -> tuple[Path, dict[str, Any], int]:
    checkpoint_path = sync_root / "weight_v000000"
    _write_hf_checkpoint_stub(checkpoint_path)
    return checkpoint_path, {
        "update_weight_mode": "full",
        "update_weight_transport": "disk",
        "update_weight_path": str(checkpoint_path),
    }, 0


def _prepare_delta_smoke_checkpoint(sync_root: Path) -> tuple[Path, dict[str, Any], int]:
    full_root = sync_root / "full"
    base = full_root / "weight_v000000"
    target = full_root / "weight_v000001"
    delta = sync_root / "weight_v000001"
    _write_hf_checkpoint_stub(base)
    target.mkdir(parents=True, exist_ok=True)
    (target / "config.json").write_text("{}", encoding="utf-8")
    (target / "model.safetensors").write_bytes(b"updated-smoke-weight")
    publish_delta_checkpoint(
        base_dir=base,
        target_dir=target,
        delta_dir=delta,
        base_version=0,
        target_version=1,
        block_size=8,
    )
    return delta, {
        "update_weight_mode": "delta",
        "update_weight_transport": "disk",
        "update_weight_path": str(delta),
        "delta_root": str(sync_root),
        "delta_initial": False,
        "base_checkpoint_path": str(base),
        "base_version_id": 0,
        "update_weight_local_checkpoint_dir": str(sync_root / "local-checkpoint"),
    }, 1


def _assert_eight_gpu_placement(statuses: list[dict[str, Any]], expected: int) -> None:
    if len(statuses) != expected:
        raise AssertionError(f"Expected {expected} actor statuses, got {len(statuses)}.")
    logical_gpu_ids = []
    for status in statuses:
        ids = status.get("accelerator_ids", {}).get("GPU") or []
        if len(ids) != 1:
            raise AssertionError(f"Expected one Ray GPU id per actor, got status={status}.")
        logical_gpu_ids.extend(str(item) for item in ids)
        if not status.get("cuda_available"):
            raise AssertionError(f"Actor CUDA is not available: {status}")
        if status.get("cuda_device_count") != 1:
            raise AssertionError(f"Actor should see exactly one CUDA device: {status}")
    if len(set(logical_gpu_ids)) != expected:
        raise AssertionError(f"Expected {expected} distinct Ray GPU ids, got {logical_gpu_ids}.")


def _assert_update_result(result: dict[str, Any], expected: int) -> None:
    if result.get("num_actors") != expected:
        raise AssertionError(f"Expected {expected} weight-sync actors, got {result.get('num_actors')}.")
    if len(result.get("preflight") or []) != expected:
        raise AssertionError(f"Expected {expected} preflight results, got {result.get('preflight')}.")
    if len(result.get("actors") or []) != expected:
        raise AssertionError(f"Expected {expected} update results, got {result.get('actors')}.")


@ray.remote(num_gpus=1)
class WeightSyncSmokeActor:
    def __init__(self, index: int) -> None:
        import ray
        import torch

        self.index = index
        self.pid = os.getpid()
        self.accelerator_ids = ray.get_runtime_context().get_accelerator_ids()
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            self.cuda_probe = float(torch.ones(1, device="cuda").item())
            self.cuda_device_count = torch.cuda.device_count()
            self.cuda_device_name = torch.cuda.get_device_name(0)
        else:
            self.cuda_probe = None
            self.cuda_device_count = 0
            self.cuda_device_name = None

    def validate_weight_path(self, checkpoint_path: str, require_hf_checkpoint: bool = True) -> dict[str, Any]:
        path = Path(checkpoint_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Smoke checkpoint is not visible to actor {self.index}: {path}")
        if not path.is_dir():
            raise FileNotFoundError(f"Smoke checkpoint is not a directory: {path}")
        has_config = (path / "config.json").exists()
        num_weights = len(list(path.glob("*.safetensors"))) + len(list(path.glob("*.bin")))
        if require_hf_checkpoint and not has_config:
            raise FileNotFoundError(f"Smoke checkpoint is missing config.json: {path}")
        if require_hf_checkpoint and num_weights < 1:
            raise FileNotFoundError(f"Smoke checkpoint has no weight files: {path}")
        return {
            "actor_index": self.index,
            "pid": self.pid,
            "resolved_path": str(path),
            "has_config": has_config,
            "num_weights": num_weights,
            "accelerator_ids": self.accelerator_ids,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }

    def validate_weight_update(self, **payload: Any) -> dict[str, Any]:
        metadata = dict(payload.get("metadata") or {})
        if metadata.get("update_weight_mode") == "delta":
            from lmms_engine.rl.training_engine.disk_delta import validate_delta_checkpoint

            return {
                "actor_index": self.index,
                "pid": self.pid,
                "mode": "delta",
                "base": self.validate_weight_path(metadata["base_checkpoint_path"], require_hf_checkpoint=True),
                "delta": validate_delta_checkpoint(payload["checkpoint_path"]),
            }
        return self.validate_weight_path(
            payload["checkpoint_path"],
            require_hf_checkpoint=payload.get("require_hf_checkpoint", True),
        )

    def update_weights(self, **payload: Any) -> dict[str, Any]:
        metadata = dict(payload.get("metadata") or {})
        if metadata.get("update_weight_mode") == "delta":
            from lmms_engine.rl.training_engine.disk_delta import (
                apply_delta_checkpoints,
                init_local_checkpoint,
                local_checkpoint_state,
            )

            local_dir = metadata["update_weight_local_checkpoint_dir"]
            init_local_checkpoint(
                local_dir=local_dir,
                base_dir=metadata["base_checkpoint_path"],
                base_version=int(metadata["base_version_id"]),
            )
            apply_delta_checkpoints(
                local_dir=local_dir,
                delta_root=metadata["delta_root"],
                target_version=int(payload["version_id"]),
            )
            validation = self.validate_weight_path(local_dir, require_hf_checkpoint=True)
            validation["local_state"] = local_checkpoint_state(local_dir)
        else:
            validation = self.validate_weight_path(
                payload["checkpoint_path"],
                require_hf_checkpoint=payload.get("require_hf_checkpoint", True),
            )
        return {
            "actor_index": self.index,
            "pid": self.pid,
            "version_id": payload.get("version_id"),
            "validation": validation,
            "metadata": dict(payload.get("metadata") or {}),
            "reset_prefix_cache": payload.get("reset_prefix_cache"),
        }

    def status(self) -> dict[str, Any]:
        return {
            "actor_index": self.index,
            "pid": self.pid,
            "accelerator_ids": self.accelerator_ids,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_available": self.cuda_probe == 1.0,
            "cuda_device_count": self.cuda_device_count,
            "cuda_device_name": self.cuda_device_name,
        }


if __name__ == "__main__":
    main()
