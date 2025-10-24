import torch
import torch.distributed as dist
from transformers import Qwen3MoeForCausalLM, Qwen3MoeConfig
import os
from torch.distributed.device_mesh import DeviceMesh
from lmms_engine.parallel.process_group_manager import setup_process_group_manager
import lmms_engine.parallel.process_group_manager as pgm
from lmms_engine.parallel.plan.qwen3_moe import apply_qwen3_moe_parallel
from lmms_engine.models.qwen3_moe import apply_liger_kernel_to_qwen3_moe

torch.manual_seed(42)
dist.init_process_group(backend="nccl", init_method="env://")
world_size=int(os.environ["WORLD_SIZE"])
local_rank=int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
setup_process_group_manager(tp_size=1, cp_size=1, pp_size=1, dp_size=1, ep_size=world_size)
config = Qwen3MoeConfig(
    num_hidden_layers=1,
)
model = Qwen3MoeForCausalLM(config).to(torch.float16).to("cuda")
apply_liger_kernel_to_qwen3_moe(model)
apply_qwen3_moe_parallel(model, ep_mesh=pgm.process_group_manager.device_mesh["ep"], tp_mesh=None)

if local_rank == 0:
    input_ids = [1,2,3,4]
else:
    input_ids = [2,3,4,5]
attention_mask = [1,1,1,1]
outputs = model(input_ids=torch.tensor([input_ids]).cuda(), attention_mask=torch.tensor([attention_mask]).cuda())
print("Logits", outputs.logits)