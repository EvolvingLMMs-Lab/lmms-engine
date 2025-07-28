import torch


def flops_per_token(config):
    num_hidden_layers = config.num_hidden_layers
    hidden_size = config.hidden_size
    vocab_size = config.vocab_size
    intermediate_size = config.intermediate_size
    kv_heads = config.num_key_value_heads
    attn_heads = config.num_attention_heads
    total_params = 0.0  # initilize total_params as float, to avoid overflow of 'flops' when using 512 gpus
    # head, mebedding not considered
    total_params += vocab_size * hidden_size
    # transformers
    params_per_block = 2 * hidden_size * hidden_size
    params_per_block += 4 * hidden_size * hidden_size * kv_heads // attn_heads
    params_per_block += 3 * hidden_size * intermediate_size
    total_params += params_per_block * num_hidden_layers

    flops = 6 * total_params
    return flops


def calc_gpt_flops(attention_mask, config):
    tokens_count = torch.sum(attention_mask != 0).item()
    flops = flops_per_token(config) * tokens_count
    token_count_list = torch.sum(attention_mask != 0, dim=1).tolist()
    for seq_len in token_count_list:
        flops += 12 * seq_len * seq_len * config.num_hidden_layers * config.hidden_size
    return flops
