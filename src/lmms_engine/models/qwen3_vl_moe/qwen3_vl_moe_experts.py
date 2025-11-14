import torch
import torch.nn as nn
from torch.distributed._tensor import DTensor


class Qwen3VLMoeExperts(nn.Module):
    def __init__(
        self,
        num_experts: int,
        hidden_dim: int,
        intermediate_size: int,
        act_fn: nn.Module,
    ):
        super().__init__()
        # fused gate_up_proj
        self.gate_up_proj = nn.Parameter(
            torch.empty(num_experts, hidden_dim, 2 * intermediate_size),
            requires_grad=True,
        )
        self.down_proj = nn.Parameter(
            torch.empty(num_experts, intermediate_size, hidden_dim),
            requires_grad=True,
        )
        self.num_experts = num_experts
        self.act_fn = act_fn

    def forward(self, *routed_input):
        out_experts_split = []
        if isinstance(self.down_proj, DTensor):
            down_proj = self.down_proj.to_local()
            gate_up_proj = self.gate_up_proj.to_local()
        else:
            down_proj = self.down_proj
            gate_up_proj = self.gate_up_proj

        for idx, x in enumerate(routed_input):
            gate_up = torch.matmul(x, gate_up_proj[idx])
            gate, up = gate_up.chunk(2, dim=-1)
            hidden = up * self.act_fn(gate)
            hidden = torch.matmul(hidden, down_proj[idx])
            out_experts_split.append(hidden)

        return torch.cat(out_experts_split, dim=0)
