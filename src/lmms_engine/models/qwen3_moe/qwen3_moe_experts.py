import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor


class Qwen3MoeExperts(nn.Module):
    def __init__(self, num_experts: int, hidden_dim: int, intermediate_size: int, act_fn):
        super().__init__()
        self.gate_proj = torch.nn.Parameter(
            torch.empty(num_experts, intermediate_size, hidden_dim),
            requires_grad=True,
        )
        self.up_proj = torch.nn.Parameter(
            torch.empty(num_experts, intermediate_size, hidden_dim),
            requires_grad=True,
        )
        self.down_proj = torch.nn.Parameter(
            torch.empty(num_experts, hidden_dim, intermediate_size),
            requires_grad=True,
        )
        self.num_experts = num_experts
        self.act_fn = act_fn

    def forward(self, *routed_input):
        out_experts_split = []
        if isinstance(self.down_proj, DTensor):
            down_proj = self.down_proj.to_local()
            up_proj = self.up_proj.to_local()
            gate_proj = self.gate_proj.to_local()
        else:
            down_proj = self.down_proj
            up_proj = self.up_proj
            gate_proj = self.gate_proj

        for idx, x in enumerate(routed_input):
            hidden = self.act_fn(torch.matmul(x, gate_proj[idx].transpose(-2, -1)))
            hidden = hidden * torch.matmul(x, up_proj[idx].transpose(-2, -1))
            hidden = torch.matmul(hidden, down_proj[idx].transpose(-2, -1))
            out_experts_split.append(hidden)
        return torch.cat(out_experts_split, dim=0)
