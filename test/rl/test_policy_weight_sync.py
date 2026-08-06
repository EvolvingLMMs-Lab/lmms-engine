import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lmms_engine.rl.model_server.vllm import VLLMChatModelServer
from lmms_engine.rl.protocol import ModelVersion
from lmms_engine.rl.training_engine import disk_delta
from lmms_engine.rl.training_engine.disk_delta import (
    apply_delta_checkpoints,
    init_local_checkpoint,
    local_checkpoint_state,
    publish_delta_checkpoint,
)
from lmms_engine.rl.training_engine.weight_sync import RayActorWeightSyncClient


class _FakeLLM:
    def __init__(self):
        self.calls = []
        self.prefix_cache_reset = False

    def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        self.calls.append(
            {
                "method": method,
                "timeout": timeout,
                "args": args,
                "kwargs": dict(kwargs or {}),
            }
        )
        return []

    def reset_prefix_cache(self):
        self.prefix_cache_reset = True
        return True


class _RemoteUpdateWeights:
    def __init__(self, actor):
        self.actor = actor

    def remote(self, **kwargs):
        self.actor.payloads.append(kwargs)
        return {"actor": self.actor.name, "version_id": kwargs["version_id"]}


class _RemoteValidateWeightPath:
    def __init__(self, actor):
        self.actor = actor

    def remote(self, checkpoint_path, require_hf_checkpoint=True):
        self.actor.validations.append(
            {
                "checkpoint_path": checkpoint_path,
                "require_hf_checkpoint": require_hf_checkpoint,
            }
        )
        return {"actor": self.actor.name, "resolved_path": checkpoint_path}


class _RemoteValidateWeightUpdate:
    def __init__(self, actor):
        self.actor = actor

    def remote(self, **kwargs):
        self.actor.validations.append(kwargs)
        return {"actor": self.actor.name, "checkpoint_path": kwargs["checkpoint_path"]}


class _FakeActor:
    def __init__(self, name):
        self.name = name
        self.payloads = []
        self.validations = []
        self.update_weights = _RemoteUpdateWeights(self)
        self.validate_weight_path = _RemoteValidateWeightPath(self)
        self.validate_weight_update = _RemoteValidateWeightUpdate(self)


class _FakeRay:
    def __init__(self):
        self.timeout = None

    def get(self, refs, timeout=None):
        self.timeout = timeout
        return refs


class TestPolicyWeightSync(unittest.TestCase):
    def test_vllm_model_server_reloads_checkpoint_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_hf_checkpoint_stub(tmpdir)
            server = VLLMChatModelServer.__new__(VLLMChatModelServer)
            server.llm = _FakeLLM()
            server.weight_version_id = None
            server.weight_checkpoint_path = None

            result = server.update_weights(
                checkpoint_path=tmpdir,
                version_id=3,
                metadata={"source": "unit"},
                timeout_s=12,
            )

        self.assertEqual(result["version_id"], 3)
        self.assertEqual(server.weight_version_id, 3)
        self.assertEqual(server.llm.calls[0]["method"], "reload_weights")
        self.assertEqual(server.llm.calls[0]["timeout"], 12)
        self.assertEqual(server.llm.calls[0]["kwargs"]["weights_path"], str(Path(tmpdir).resolve()))
        self.assertTrue(server.llm.calls[0]["kwargs"]["is_checkpoint_format"])
        self.assertTrue(server.llm.prefix_cache_reset)

    def test_vllm_delta_local_checkpoint_dir_is_reused(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base = root / "full" / "weight_v000000"
            target = root / "full" / "weight_v000001"
            delta = root / "weight_v000001"
            _write_hf_checkpoint_stub(str(base), weight_bytes=b"base")
            _write_hf_checkpoint_stub(str(target), weight_bytes=b"target")
            publish_delta_checkpoint(
                base_dir=base,
                target_dir=target,
                delta_dir=delta,
                base_version=0,
                target_version=1,
                block_size=2,
            )

            server = VLLMChatModelServer.__new__(VLLMChatModelServer)
            server.llm = _FakeLLM()
            server.weight_version_id = None
            server.weight_checkpoint_path = "/source/model"
            server._delta_local_checkpoint_dir = None

            server.update_weights(
                checkpoint_path=str(base),
                version_id=0,
                metadata={
                    "update_weight_mode": "delta",
                    "update_weight_transport": "disk",
                    "delta_initial": True,
                    "base_checkpoint_path": str(base),
                    "base_version_id": 0,
                },
                reset_prefix_cache=False,
            )
            first_local_dir = Path(server._delta_local_checkpoint_dir)

            server.update_weights(
                checkpoint_path=str(delta),
                version_id=1,
                metadata={
                    "update_weight_mode": "delta",
                    "update_weight_transport": "disk",
                    "delta_initial": False,
                    "base_checkpoint_path": str(base),
                    "base_version_id": 0,
                    "delta_root": str(root),
                },
                reset_prefix_cache=False,
            )

            self.assertEqual(Path(server._delta_local_checkpoint_dir), first_local_dir)
            self.assertEqual(Path(server.llm.calls[0]["kwargs"]["weights_path"]), first_local_dir)
            self.assertEqual(Path(server.llm.calls[1]["kwargs"]["weights_path"]), first_local_dir)
            self.assertEqual((first_local_dir / "model.safetensors").read_bytes(), b"target")
            self.assertEqual(local_checkpoint_state(first_local_dir)["version"], 1)

    def test_ray_actor_weight_sync_updates_all_actors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_hf_checkpoint_stub(tmpdir)
            ray = _FakeRay()
            actors = [_FakeActor("a0"), _FakeActor("a1")]
            client = RayActorWeightSyncClient(
                actor_handles=actors,
                timeout_s=45,
                extra_kwargs={"reset_prefix_cache": False},
            )

            with patch("lmms_engine.rl.training_engine.weight_sync._require_ray", return_value=ray):
                result = client.reload_weights(
                    ModelVersion(version_id=7, checkpoint_path=tmpdir, metadata={"kind": "test"})
                )

        self.assertEqual(ray.timeout, 45)
        self.assertEqual(result["num_actors"], 2)
        self.assertEqual(result["version_id"], 7)
        self.assertEqual(len(result["preflight"]), 2)
        for actor in actors:
            self.assertEqual(actor.validations[0]["checkpoint_path"], tmpdir)
            self.assertTrue(actor.validations[0]["require_hf_checkpoint"])
            self.assertEqual(actor.payloads[0]["checkpoint_path"], tmpdir)
            self.assertEqual(actor.payloads[0]["version_id"], 7)
            self.assertEqual(actor.payloads[0]["metadata"], {"kind": "test"})
            self.assertTrue(actor.payloads[0]["require_hf_checkpoint"])
            self.assertFalse(actor.payloads[0]["reset_prefix_cache"])

    def test_ray_actor_weight_sync_rejects_missing_local_checkpoint(self):
        client = RayActorWeightSyncClient(actor_handles=[_FakeActor("a0")])
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_path = str(Path(tmpdir) / "missing-policy")
            with self.assertRaises(FileNotFoundError):
                client.reload_weights(ModelVersion(version_id=1, checkpoint_path=missing_path))

    def test_disk_delta_applies_patch_to_local_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base = root / "base"
            target = root / "target"
            local = root / "local"
            delta = root / "weight_v000010"
            _write_hf_checkpoint_stub(str(base), weight_bytes=b"abc" * 1024)
            _write_hf_checkpoint_stub(str(target), weight_bytes=b"abc" * 512 + b"xyz" * 512)

            publish_delta_checkpoint(
                base_dir=base,
                target_dir=target,
                delta_dir=delta,
                base_version=0,
                target_version=10,
                block_size=128,
            )
            init_local_checkpoint(local_dir=local, base_dir=base, base_version=0)
            apply_delta_checkpoints(local_dir=local, delta_root=root, target_version=10)

            self.assertEqual((local / "model.safetensors").read_bytes(), (target / "model.safetensors").read_bytes())
            self.assertEqual(local_checkpoint_state(local)["version"], 10)

    def test_disk_delta_lock_retries_transient_busy_lock(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base = root / "base"
            local = root / "local"
            _write_hf_checkpoint_stub(str(base), weight_bytes=b"abc")
            calls = {"exclusive": 0}

            def flaky_flock(handle, operation):
                if operation & disk_delta.fcntl.LOCK_EX:
                    calls["exclusive"] += 1
                    if calls["exclusive"] <= 2:
                        raise BlockingIOError(11, "Resource temporarily unavailable")
                return None

            with patch("lmms_engine.rl.training_engine.disk_delta.fcntl.flock", side_effect=flaky_flock), patch(
                "lmms_engine.rl.training_engine.disk_delta.time.sleep", return_value=None
            ):
                init_local_checkpoint(local_dir=local, base_dir=base, base_version=0)

            self.assertGreaterEqual(calls["exclusive"], 3)
            self.assertEqual((local / "model.safetensors").read_bytes(), b"abc")
            self.assertEqual(local_checkpoint_state(local)["version"], 0)

    def test_disk_delta_initial_sync_resets_newer_local_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base = root / "base"
            target = root / "target"
            local = root / "local"
            delta = root / "weight_v000001"
            _write_hf_checkpoint_stub(str(base), weight_bytes=b"base")
            _write_hf_checkpoint_stub(str(target), weight_bytes=b"target")
            publish_delta_checkpoint(
                base_dir=base,
                target_dir=target,
                delta_dir=delta,
                base_version=0,
                target_version=1,
                block_size=2,
            )
            init_local_checkpoint(local_dir=local, base_dir=base, base_version=0)
            apply_delta_checkpoints(local_dir=local, delta_root=root, target_version=1)

            init_local_checkpoint(local_dir=local, base_dir=base, base_version=0, reset_if_newer=True)

            self.assertEqual((local / "model.safetensors").read_bytes(), b"base")
            self.assertEqual(local_checkpoint_state(local)["version"], 0)


def _write_hf_checkpoint_stub(path: str, weight_bytes: bytes = b"stub") -> None:
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text("{}", encoding="utf-8")
    (root / "model.safetensors").write_bytes(weight_bytes)


if __name__ == "__main__":
    unittest.main()
