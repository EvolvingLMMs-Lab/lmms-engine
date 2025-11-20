import math
from typing import List, Optional, Tuple, Union, Iterable

from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteSchedulerOutput, FlowMatchEulerDiscreteScheduler

import torch
from loguru import logger
from transformers import PretrainedConfig

from lmms_engine.utils import TrainUtilities

# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


VALID_CONFIG_TYPE = {
    "llama",
    "qwen2",
    "qwen2_vl",
    "qwen2_5_vl",
    "qwen2_5_omni",
    "qwen2_5_omni_thinker",
    "qwen3",
    "qwen3_dllm",
    "qwen3_moe",
    "qwen3_vl",
    "deepseek_v3",
    "minicpmv",
    "minicpmo",
    "llava_onevision",
}


class FlopsCounter:
    """
    Used to count mfu during training loop

    Example:
        flops_counter = FlopsCounter(config)
        flops_achieved, flops_promised = flops_counter.estimate_flops(tokens_list, delta_time)

    """

    def __init__(self, config: PretrainedConfig):
        if config.model_type not in VALID_CONFIG_TYPE:
            logger.warning(
                f"Only support config type of {VALID_CONFIG_TYPE}, but got {config.model_type}. MFU will not be counted."
            )

        self.estimate_func = {
            "qwen2": self._estimate_qwen2_flops,
            "llama": self._estimate_qwen2_flops,
            "qwen2_moe": self._estimate_qwen2_moe_flops,
            "qwen2_vl": self._estimate_qwen2_flops,
            "qwen2_5_vl": self._estimate_qwen2_flops,
            "qwen2_5_omni": self._estimate_qwen2_flops,
            "qwen2_5_omni_thinker": self._estimate_qwen2_flops,
            "qwen3": self._estimate_qwen2_flops,
            "qwen3_dllm": self._estimate_qwen2_flops,
            "qwen3_moe": self._estimate_qwen2_moe_flops,
            "qwen3_vl": self._estimate_qwen2_flops,
            "deepseek_v3": self._estimate_deepseek_v3_flops,
            "minicpmv": self._estimate_qwen2_flops,
            "minicpmo": self._estimate_qwen2_flops,
            "llava_onevision": self._estimate_qwen2_flops,
            "bagel": self._estimate_qwen2_flops,
        }
        if config.model_type in [
            "llava_onevision",
            "qwen3_vl",
            "qwen2_5_omni",
            "qwen2_5_omni_thinker",
        ]:
            self.config = config.text_config
            self.config.model_type = config.model_type
        elif config.model_type == "bagel":
            self.config = config.llm_config
        else:
            self.config = config

    def _estimate_unknown_flops(self, tokens_sum, batch_seqlens, delta_time):
        return 0

    def _estimate_qwen2_flops(self, tokens_sum, batch_seqlens, delta_time):
        config = self.config
        hidden_size = config.hidden_size
        vocab_size = config.vocab_size
        num_hidden_layers = config.num_hidden_layers
        num_key_value_heads = config.num_key_value_heads
        num_attention_heads = config.num_attention_heads
        intermediate_size = config.intermediate_size

        head_dim = getattr(
            config,
            "head_dim",
            config.hidden_size // config.num_attention_heads,
        )
        q_size = num_attention_heads * head_dim
        k_size = num_key_value_heads * head_dim
        v_size = num_key_value_heads * head_dim

        # non-attn per layer parm
        # Qwen2/LLama use SwiGelu, gate, having up and down linear layer in mlp
        mlp_N = hidden_size * intermediate_size * 3
        attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
        emd_and_lm_head_N = vocab_size * hidden_size * 2
        # non-attn all_layer parm
        dense_N = (mlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N
        # non-attn all_layer & all_token fwd & bwd flops
        dense_N_flops = 6 * dense_N * tokens_sum

        # attn all_layer & all_token fwd & bwd flops
        seqlen_square_sum = 0
        for seqlen in batch_seqlens:
            seqlen_square_sum += seqlen * seqlen
        attn_qkv_flops = 12 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers

        # all_layer & all_token fwd & bwd flops
        flops_all_token = dense_N_flops + attn_qkv_flops
        flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
        return flops_achieved

    def _estimate_deepseek_v3_flops(self, tokens_sum, batch_seqlens, delta_time):
        hidden_size = self.config.hidden_size
        vocab_size = self.config.vocab_size
        moe_intermediate_size = self.config.moe_intermediate_size
        num_hidden_layers = self.config.num_hidden_layers
        first_k_dense_replace = self.config.first_k_dense_replace
        num_query_heads = self.config.num_attention_heads
        moe_num_expert = self.config.n_routed_experts

        moe_topk = self.config.num_experts_per_tok
        share_expert_num = self.config.n_shared_experts

        # non-attn per layer parm
        moe_gata_N = hidden_size * moe_num_expert
        # moe has fc1_1, fc1_2 and fc2 using SwiGLU in ExpertMlp layer & shared experts
        moe_expertmlp_N = hidden_size * moe_intermediate_size * (moe_topk + share_expert_num) * 3
        # MLA attn
        attn_linear_N = 0
        q_head_dim = self.config.qk_nope_head_dim + self.config.qk_rope_head_dim
        if self.config.q_lora_rank is None:
            attn_linear_N += hidden_size * num_query_heads * q_head_dim
        else:
            attn_linear_N += hidden_size * self.config.q_lora_rank
            attn_linear_N += num_query_heads * q_head_dim * self.config.q_lora_rank

        attn_linear_N += hidden_size * (self.config.kv_lora_rank + self.config.qk_rope_head_dim)
        attn_linear_N += (
            num_query_heads
            * (q_head_dim - self.config.qk_rope_head_dim + self.config.v_head_dim)
            * self.config.kv_lora_rank
        )
        attn_linear_N += num_query_heads * self.config.v_head_dim * hidden_size
        emd_and_lm_head_N = vocab_size * hidden_size * 2
        # non-attn all_layer parm
        moe_N = (
            (moe_gata_N + moe_expertmlp_N + attn_linear_N) * (num_hidden_layers - first_k_dense_replace)
            + (hidden_size * self.config.intermediate_size * 3 + attn_linear_N) * first_k_dense_replace
            + emd_and_lm_head_N
        )
        # non-attn all_layer & all_token fwd & bwd flops
        dense_N_flops = 6 * moe_N * tokens_sum

        # attn all_layer & all_token fwd & bwd flops
        seqlen_square_sum = 0
        for seqlen in batch_seqlens:
            seqlen_square_sum += seqlen * seqlen * num_hidden_layers

        attn_qkv_flops = 12 * seqlen_square_sum * q_head_dim * num_query_heads
        # all_layer & all_token fwd & bwk flops
        flops_all_token = dense_N_flops + attn_qkv_flops
        flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12

        return flops_achieved

    def _estimate_qwen2_moe_flops(self, tokens_sum, batch_seqlens, delta_time):
        hidden_size = self.config.hidden_size
        vocab_size = self.config.vocab_size
        num_hidden_layers = self.config.num_hidden_layers
        num_key_value_heads = self.config.num_key_value_heads
        num_attention_heads = self.config.num_attention_heads
        moe_intermediate_size = self.config.moe_intermediate_size
        moe_topk = self.config.num_experts_per_tok
        num_experts = self.config.num_experts

        head_dim = getattr(
            self.config,
            "head_dim",
            self.config.hidden_size // self.config.num_attention_heads,
        )
        q_size = num_attention_heads * head_dim
        k_size = num_key_value_heads * head_dim
        v_size = num_key_value_heads * head_dim

        # non-attn per layer parm
        # gate + moe export
        moe_mlp_N = hidden_size * moe_topk * moe_intermediate_size * 3 + hidden_size * num_experts
        attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
        emd_and_lm_head_N = vocab_size * hidden_size * 2
        # non-attn all_layer parm
        dense_N = (moe_mlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N
        # non-attn all_layer & all_token fwd & bwd flops
        dense_N_flops = 6 * dense_N * tokens_sum

        # attn all_layer & all_token fwd & bwd flops
        seqlen_square_sum = 0
        for seqlen in batch_seqlens:
            seqlen_square_sum += seqlen * seqlen
        attn_qkv_flops = 12 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers

        # all_layer & all_token fwd & bwd flops
        flops_all_token = dense_N_flops + attn_qkv_flops
        flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
        return flops_achieved

    def estimate_flops(self, batch_seqlens, delta_time):
        """
        Estimate the FLOPS based on the number of valid tokens in the current batch and the time taken.

        Args:
            batch_seqlens (List[int]): A list where each element represents the number of valid tokens in the
                current batch.
            delta_time (float): The time taken to process the batch, in seconds.

        Returns:
            estimated_flops (float): The estimated FLOPS based on the input tokens and time.
            promised_flops (float): The expected FLOPS of the current device.
        """
        # Flatten batch_seqlens if it contains nested lists (common for vision models)
        if isinstance(batch_seqlens, (list, tuple)):
            flat_seqlens = []
            for item in batch_seqlens:
                if isinstance(item, (list, tuple)):
                    flat_seqlens.extend(item)
                else:
                    flat_seqlens.append(item)
            batch_seqlens = flat_seqlens
        tokens_sum = sum(batch_seqlens)
        func = self.estimate_func.get(self.config.model_type, self._estimate_unknown_flops)
        estimated_flops = func(tokens_sum, batch_seqlens, delta_time)
        promised_flops = TrainUtilities.get_device_flops()
        return estimated_flops, promised_flops


def setup_flops_counter(config: PretrainedConfig) -> FlopsCounter:
    """
    Setup the FlopsCounter based on the provided configuration.

    Args:
        config (PretrainedConfig): The configuration object for the model.

    Returns:
        FlopsCounter: An instance of FlopsCounter initialized with the given configuration.
    """
    global flops_counter
    flops_counter = FlopsCounter(config)


def sde_step_with_logprob(
    scheduler: FlowMatchEulerDiscreteScheduler,
    model_output: torch.FloatTensor,
    timestep: Optional[Union[float, torch.FloatTensor]] = None,
    sample: Optional[torch.FloatTensor] = None,
    prev_sample: Optional[torch.FloatTensor] = None,
    generator: Optional[torch.Generator] = None,
) -> Union[FlowMatchEulerDiscreteSchedulerOutput, Tuple]:
    """
    Predict the sample from the previous timestep by reversing the SDE. This function propagates the flow
    process from the learned model outputs (most often the predicted velocity).

    Args:
        model_output (`torch.FloatTensor`):
            The direct output from learned flow model.
        timestep (`float`):
            The current discrete timestep in the diffusion chain.
        sample (`torch.FloatTensor`):
            A current instance of a sample created by the diffusion process.
        generator (`torch.Generator`, *optional*):
            A random number generator.
    """
    assert timestep is not None or sigmas is not None, "timestep or sigmas must be provided"
    view_shape = (-1,) + (1,) * (model_output.ndim - 1)
    step_index = [scheduler.index_for_timestep(t) for t in timestep]
    prev_step_index = [step+1 for step in step_index]
    # Dynamically determine view shape based on model_output dimensions
    # e.g., 2D -> (-1, 1), 3D -> (-1, 1, 1), 4D -> (-1, 1, 1, 1)
    sigma = scheduler.sigmas[step_index].view(*view_shape).to(model_output.device)
    sigma_prev = scheduler.sigmas[prev_step_index].view(*view_shape).to(model_output.device)
    sigma_max = scheduler.sigmas[1].item()
    dt = sigma_prev - sigma

    std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma)))*0.7
    # our sde
    # print(sample.shape, model_output.shape, std_dev_t.shape, sigma.shape, dt.shape)
    prev_sample_mean = sample*(1+std_dev_t**2/(2*sigma)*dt) + model_output * (1+std_dev_t**2*(1-sigma)/(2*sigma))*dt
    
    if prev_sample is not None and generator is not None:
        raise ValueError(
            "Cannot pass both generator and prev_sample. Please make sure that either `generator` or"
            " `prev_sample` stays `None`."
        )

    if prev_sample is None:
        variance_noise = randn_tensor(
            model_output.shape,
            generator=generator,
            device=model_output.device,
            dtype=model_output.dtype,
        )
        prev_sample = prev_sample_mean + std_dev_t * torch.sqrt(-1*dt) * variance_noise


    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * ((std_dev_t * torch.sqrt(-1*dt))**2))
        - torch.log(std_dev_t * torch.sqrt(-1*dt))
        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
    )

    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
    
    return prev_sample, log_prob, prev_sample_mean, std_dev_t * torch.sqrt(-1*dt)


def sde_step_with_logprob_simple(
    scheduler: FlowMatchEulerDiscreteScheduler,
    model_output: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    sample: torch.FloatTensor,
    prev_sample: Optional[torch.FloatTensor] = None,
    generator: Optional[torch.Generator] = None,
):
    """
    Predict the sample from the previous timestep by reversing the SDE. This function propagates the flow
    process from the learned model outputs (most often the predicted velocity).

    Args:
        model_output (`torch.FloatTensor`):
            The direct output from learned flow model.
        timestep (`float`):
            The current discrete timestep in the diffusion chain.
        sample (`torch.FloatTensor`):
            A current instance of a sample created by the diffusion process.
        generator (`torch.Generator`, *optional*):
            A random number generator.
    """

    step_index = [scheduler.index_for_timestep(t) for t in timestep]
    prev_step_index = [step+1 for step in step_index]

    # Dynamically determine view shape based on model_output dimensions
    # e.g., 2D -> (-1, 1), 3D -> (-1, 1, 1), 4D -> (-1, 1, 1, 1)
    view_shape = (-1,) + (1,) * (model_output.ndim - 1)

    sigma = scheduler.sigmas[step_index].view(*view_shape).to(model_output.device)
    sigma_prev = scheduler.sigmas[prev_step_index].view(*view_shape).to(model_output.device)
    sigma_max = scheduler.sigmas[1].item()
    dt = sigma_prev - sigma


    eta = 0.5
    Dt = - dt * eta

    prev_sample_mean = sample * (1 - Dt / (1 - torch.where(sigma == 1, sigma_max, sigma))) + model_output * (dt - Dt)
    
    std_dev_t = torch.sqrt(2 * Dt * (sigma / (1 - torch.where(sigma == 1, sigma_max, sigma))))
    
    if prev_sample is not None and generator is not None:
        raise ValueError(
            "Cannot pass both generator and prev_sample. Please make sure that either `generator` or"
            " `prev_sample` stays `None`."
        )

    if prev_sample is None:
        # Generate noise if not provided
        variance_noise = randn_tensor(
            model_output.shape,
            generator=generator,
            device=model_output.device,
            dtype=model_output.dtype,
        )
            
        prev_sample = prev_sample_mean + std_dev_t * variance_noise


    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * (std_dev_t**2))
        - torch.log(std_dev_t)
        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
    )

    # mean along all but batch dimension
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
    
    return prev_sample, log_prob, prev_sample_mean, std_dev_t 