import unittest

from lmms_engine.rl.model_server import (
    ModelServerManager,
    normalize_model_server_configs,
)
from lmms_engine.train.rl.runner import _validate_policy_model_server

_RAY_POLICY_SPEC = {
    "name": "ray_actor_pool",
    "server": {
        "factory": "lmms_engine.rl.model_server.vllm:VLLMChatModelServer",
        "model": "/tmp/model",
    },
}


class TestPolicyModelServerValidation(unittest.TestCase):
    def test_accepts_external_openai_compatible_policy_server(self):
        _validate_policy_model_server(
            {
                "name": "openai",
                "model": "policy",
                "base_url": "http://127.0.0.1:8000/v1",
            },
            weight_sync_backend="vllm_http",
        )

    def test_rejects_external_policy_server_with_ray_actor_weight_sync(self):
        with self.assertRaisesRegex(ValueError, "vllm.backend='vllm_http'"):
            _validate_policy_model_server(
                {
                    "name": "openai",
                    "model": "policy",
                    "base_url": "http://127.0.0.1:8000/v1",
                },
                weight_sync_backend="ray_actor_pool",
            )

    def test_accepts_ray_actor_pool_vllm_wrapper(self):
        _validate_policy_model_server(_RAY_POLICY_SPEC, weight_sync_backend="ray_actor_pool")

    def test_rejects_unknown_policy_server_backend(self):
        with self.assertRaisesRegex(ValueError, "openai.*ray_actor_pool"):
            _validate_policy_model_server({"name": "debug"}, weight_sync_backend="vllm_http")

    def test_legacy_model_server_normalizes_to_policy_role(self):
        specs = normalize_model_server_configs({}, legacy_model_server=_RAY_POLICY_SPEC)
        self.assertEqual(list(specs), ["policy"])
        self.assertEqual(specs["policy"]["name"], "ray_actor_pool")

    def test_multi_role_model_servers_are_preserved(self):
        specs = normalize_model_server_configs(
            {
                "model_servers": {
                    "policy": _RAY_POLICY_SPEC,
                    "reference": {
                        "name": "openai",
                        "model": "reference",
                        "base_url": "http://127.0.0.1:8001/v1",
                    },
                }
            }
        )

        self.assertEqual(set(specs), {"policy", "reference"})
        manager = ModelServerManager(specs)
        self.assertEqual(manager.roles(), ["policy", "reference"])
        self.assertEqual(manager.client_spec("reference")["name"], "openai")
        self.assertEqual(manager.client_spec("reference")["model"], "reference")


if __name__ == "__main__":
    unittest.main()
