import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lmms_engine.rl.model_server.vllm import VLLMChatModelServer
from lmms_engine.rl.protocol import ModelVersion
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


class _FakeActor:
    def __init__(self, name):
        self.name = name
        self.payloads = []
        self.validations = []
        self.update_weights = _RemoteUpdateWeights(self)
        self.validate_weight_path = _RemoteValidateWeightPath(self)


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


def _write_hf_checkpoint_stub(path: str) -> None:
    root = Path(path)
    (root / "config.json").write_text("{}", encoding="utf-8")
    (root / "model.safetensors").write_bytes(b"stub")


if __name__ == "__main__":
    unittest.main()
