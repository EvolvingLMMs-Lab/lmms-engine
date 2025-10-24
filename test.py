import torch
import torch.distributed as dist
from transformers import Qwen3MoeForCausalLM, Qwen3MoeConfig

from torch.distributed.device_mesh import DeviceMesh

torch.manual_seed(42)
config = Qwen3MoeConfig(
    num_hidden_layers=1,
)
model = Qwen3MoeForCausalLM(config).to(torch.float16).to("cuda")
print(model)
input_ids = [1,2,3,4]
attention_mask = [1,1,1,1]
outputs = model(input_ids=torch.tensor([input_ids]).cuda(), attention_mask=torch.tensor([attention_mask]).cuda())
print("Logits", outputs.logits)