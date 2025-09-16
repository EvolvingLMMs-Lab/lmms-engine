import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor.parallel import parallelize_module

from lmms_engine.models import MOEPARALLELPATCHER
from lmms_engine.models.moe import MoE, MoEArgs
from lmms_engine.parallel.expert_parallel.apply import apply_moe_ep_tp
from lmms_engine.parallel.expert_parallel.expert_parallel import (
    ExpertParallel,
    ExpertTensorParallel,
    ReordererSequenceParallel,
    TensorParallel,
)


class TransformerBlock(torch.nn.Module):
    def __init__(self, arg: MoEArgs):
        super().__init__()
        self.moe_enabled = True
        self.moe = MoE(arg, 10, 10)
        self.moe.init_weights(1, torch.device("cuda"))

    def forward(self, x):
        return self.moe(x)


class Model(torch.nn.Module):
    def __init__(self, arg: MoEArgs):
        super().__init__()
        self.layers = torch.nn.ModuleDict()
        self.layers["0"] = TransformerBlock(arg)

    def forward(self, x):
        for layer in self.layers.values():
            x = layer(x)
        return x


local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)

dist.init_process_group(backend="nccl")
mesh_2d = init_device_mesh("cuda", (4,), mesh_dim_names=("ep",))
# print(mesh_2d["ep"])

torch.manual_seed(5)
arg = MoEArgs()
arg.use_grouped_mm = False
model = Model(arg).cuda()
input = torch.rand((1, 1, 10)).cuda()
print(model(input))
MOEPARALLELPATCHER._apply_ep_tp(
    model=model,
    tp_mesh=None,
    ep_mesh=mesh_2d["ep"],
    ep_tp_mesh=None,
    etp_enabled=False,
)

print(model(input))
