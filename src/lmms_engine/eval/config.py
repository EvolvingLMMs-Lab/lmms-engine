from typing import Any, Dict, List, Optional, Union

from lmms_engine.protocol import Args


class EvalConfig(Args):
    server_url: Optional[str] = "http://localhost:8000"
    model: Optional[str] = "vllm"
    tasks: Optional[List[str]] = ["mmmu_val"]
    model_args: Optional[Dict[str, Any]] = {
        "model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "tensor_parallel_size": 1,
        "disable_log_stats": True,
        "gpu_memory_utilization": 0.8,
    }
    # The key to retrieve the checkpoint directory from the model_args
    checkpoint_key: Optional[str] = "model"
    num_fewshot: Optional[int] = 0
    batch_size: Optional[Union[int, str]] = 1
    device: Optional[str] = "cuda"
    limit: Optional[Union[int, float]] = None
    gen_kwargs: Optional[str] = None
    log_samples: Optional[bool] = True
    predict_only: Optional[bool] = False
    num_gpus: Optional[int] = 1

    # How often (in seconds) background thread polls the eval server for job status
    poll_interval: Optional[float] = 20.0
