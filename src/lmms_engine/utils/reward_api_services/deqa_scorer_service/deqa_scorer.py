import torch
from transformers import AutoModelForCausalLM

def load_deqascore():
    model = AutoModelForCausalLM.from_pretrained(
        "zhiyuanyou/DeQA-Score-Mix3",
        trust_remote_code=True,
        attn_implementation="eager",
        torch_dtype=torch.float16,
        device_map=None,
    ).cuda()
    model.requires_grad_(False)
    
    @torch.no_grad()
    def compute_deqascore(images):
        score = model.score(images)
        score = score / 5
        score = [sc.item() for sc in score]
        return score

    return compute_deqascore