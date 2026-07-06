import unittest

from lmms_engine.rl.ray.runtime import RayClusterSpec, RayResourcePlan


def _cluster_spec():
    return RayClusterSpec(
        num_nodes=4,
        gpus_per_node=8,
        master_addr="127.0.0.1",
        ray_port="6379",
        node_rank="0",
        train_node_rank="0",
        head_node_ip="127.0.0.1",
        wait_timeout=300,
    )


def _config(policy_replicas=16, reference_replicas=8):
    return {
        "trainer_args": {
            "rl_config": {
                "model_servers": {
                    "policy": {
                        "name": "ray_actor_pool",
                        "num_replicas": policy_replicas,
                        "actor_options": {"num_gpus": 1},
                        "server": {
                            "factory": "lmms_engine.rl.model_server.vllm:VLLMChatModelServer",
                            "model": "/tmp/policy",
                        },
                    },
                    "reference": {
                        "name": "ray_actor_pool",
                        "num_replicas": reference_replicas,
                        "actor_options": {"num_gpus": 1},
                        "server": {
                            "factory": "lmms_engine.rl.model_server.vllm:VLLMChatModelServer",
                            "model": "/tmp/reference",
                        },
                    },
                },
                "rollout": {"num_workers": 96, "batch_size": 8},
                "data_buffer": {"train_batch_size_per_gpu": 1},
                "training": {},
            }
        },
        "ray_train": {
            "num_workers": 8,
            "resources_per_worker": {"GPU": 1},
        },
    }


class TestRayResourcePlan(unittest.TestCase):
    def test_multi_role_model_servers_fit_rollout_gpus(self):
        config = _config(policy_replicas=16, reference_replicas=8)

        plan = RayResourcePlan.from_config(_cluster_spec(), config)
        plan.apply_to(config)

        self.assertEqual(plan.model_server_replicas, 24)
        self.assertEqual(plan.model_server_role_resources["policy"], (16, 1.0))
        self.assertEqual(plan.model_server_role_resources["reference"], (8, 1.0))
        self.assertEqual(
            config["trainer_args"]["rl_config"]["model_servers"]["policy"]["actor_options"]["resources"][
                "rollout_node"
            ],
            0.001,
        )
        self.assertEqual(
            config["trainer_args"]["rl_config"]["model_servers"]["reference"]["actor_options"]["resources"][
                "rollout_node"
            ],
            0.001,
        )
        self.assertEqual(
            config["trainer_args"]["rl_config"]["model_servers"]["policy"]["load_balancer_actor_options"]["resources"][
                "rollout_node"
            ],
            0.001,
        )
        self.assertEqual(
            config["trainer_args"]["rl_config"]["model_servers"]["reference"]["load_balancer_actor_options"][
                "resources"
            ]["rollout_node"],
            0.001,
        )

    def test_multi_role_model_servers_reject_gpu_overcommit(self):
        with self.assertRaisesRegex(ValueError, "Not enough rollout GPUs"):
            RayResourcePlan.from_config(_cluster_spec(), _config(policy_replicas=17, reference_replicas=8))


if __name__ == "__main__":
    unittest.main()
