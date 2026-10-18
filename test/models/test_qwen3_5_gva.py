import os
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestQwen3_5GVA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from lmms_engine.models.qwen3_5 import qwen3_5_ops
        from lmms_engine.parallel.sequence_parallel.ulysses import (
            set_ulysses_sequence_parallel_group,
        )

        if not qwen3_5_ops._HAS_FLA_GVA:
            raise unittest.SkipTest("FLA >= 0.5.0 is required")
        cls.rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(cls.rank)
        cls.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if cls.world_size > 1:
            dist.init_process_group("nccl", device_id=torch.device("cuda", cls.rank))
            set_ulysses_sequence_parallel_group(dist.group.WORLD)

    @classmethod
    def tearDownClass(cls):
        from lmms_engine.parallel.sequence_parallel.ulysses import (
            set_ulysses_sequence_parallel_group,
        )

        if cls.world_size > 1:
            set_ulysses_sequence_parallel_group(None)
            dist.destroy_process_group()

    def test_forward_and_backward_match_repeated_heads(self):
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

        from lmms_engine.models.qwen3_5 import qwen3_5_ops

        torch.manual_seed(42)
        config = Qwen3_5TextConfig(
            hidden_size=256,
            linear_num_key_heads=16,
            linear_num_value_heads=32,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            dtype=torch.bfloat16,
        )
        layer = Qwen3_5GatedDeltaNet(config, layer_idx=0).to(device="cuda", dtype=torch.bfloat16)
        torch.manual_seed(100 + self.rank)
        length = 1024
        hidden = torch.randn(1, length, config.hidden_size, device="cuda", dtype=torch.bfloat16)
        grad_output = torch.randn_like(hidden)
        cu_seqlens = torch.tensor([0, length - 127, length * self.world_size], device="cuda", dtype=torch.int32)
        kernel = qwen3_5_ops.chunk_gated_delta_rule
        results = []

        for native_gva in (False, True):
            layer.zero_grad(set_to_none=True)
            inputs = hidden.detach().requires_grad_()

            def checked_kernel(q, k, v, **kwargs):
                self.assertEqual(q.shape[2], 16 if native_gva else 32)
                self.assertEqual(k.shape[2], q.shape[2])
                self.assertEqual(v.shape[2], 32)
                if self.world_size > 1:
                    self.assertIsNotNone(kwargs.get("cp_context"))
                return kernel(q, k, v, **kwargs)

            with patch.object(qwen3_5_ops, "_HAS_FLA_GVA", native_gva):
                with patch.object(qwen3_5_ops, "chunk_gated_delta_rule", checked_kernel):
                    output = qwen3_5_ops.linear_attn_forward(layer, inputs, cu_seq_lens=cu_seqlens)
                    output.backward(grad_output)
            results.append(
                {
                    "output": output.detach().clone(),
                    "input_grad": inputs.grad.detach().clone(),
                    **{name: param.grad.detach().clone() for name, param in layer.named_parameters()},
                }
            )

        reference, actual = results
        torch.testing.assert_close(actual["output"], reference["output"], rtol=0, atol=0)
        for name in reference:
            with self.subTest(tensor=name):
                self.assertTrue(torch.isfinite(actual[name]).all().item())
                error = (actual[name].float() - reference[name].float()).norm()
                scale = reference[name].float().norm().clamp_min(1e-8)
                self.assertLess((error / scale).item(), 0.02)


if __name__ == "__main__":
    unittest.main()
