# DLLM Training with Muon Optimizer

This guide demonstrates how to train a DLLM (Differential Language Model) using the Muon/Adam optimizers in the LMMs Engine framework.

## Overview

**DLLM (Differential Language Model)** is a novel architecture that processes sequences using differential mechanisms for improved training efficiency and performance. 

let $\boldsymbol{x}_t=(x^{1}_t,\dots,x^{n}_t)$ indicate the target textual sequence,

- **upper index $i$**：representing the position of the token within a sequence；
- **lower index  $t\in[0,1]$**：denoting the denoising step
    - $t=0$：original real text $\boldsymbol{x}_{0}$;
    - $t=1$：all masked sequence $\boldsymbol{x}_{1}$;

As a result，$x_{t}^{i}$ is「the *i-th* token of the sequence at the *t-th* denoising step」. We refer to the transformation of a sequence from the fully-masked state $x_1$ to the complete text sequence $x_0$ as **denoising**; the reverse direction is termed **noising**.

We use the objective as follows: 

$$ -\log p_{\boldsymbol{\theta}}(\boldsymbol{x}) \leq \int_0^1 \frac{1}{t} \mathbb{E}_{q_{t \mid 0}\left(\boldsymbol{x}_t \mid \boldsymbol{x}_0\right)}\left[\sum_{i: \boldsymbol{x}_0^i=[\mathrm{MASK}]}-\log p_{\boldsymbol{\theta}}\left(\boldsymbol{x}_0^i \mid \boldsymbol{x}_t\right)\right] \mathrm{d} t:=\mathcal{L}_{\mathrm{MDM}} $$

which upper-bounds the negative log-likelihood of the model. 

In practice, we samples $t\sim\text{Unif}[0,1]$ and, for each fixed $t$, generates randomly masked versions of $x_t$.

**Muon Optimizer** is an advanced optimizer for large language models that improves convergence and training stability over AdamW. Our framework includes both single-GPU and multi-GPU (FSDP2) implementations.

## Quick Start

### Configuration

We provide three example configurations for training DLLMs on the FineWeb-Edu dataset using the model adapted from Qwen3 (we name it as qwen3-dllm). If you wish to train a DLLM with a custom model or dataset, you can readily adapt the provided examples.

The training configuration is defined in YAML format. Reference configuration: 

1. ```examples/vanila_dllm/dllm_train_adam_multi_gpu_deepspeed.yaml```

2. ```examples/vanila_dllm/dllm_train_muon_multi_gpu_fsdp2.yaml```

3. ```examples/vanila_dllm/dllm_train_muon_single_gpu.yaml```


Next, Key configuration highlights:

```yaml
- type: trainer
  config:
    trainer_type: dllm_trainer  # Use DLLM-specific trainer

    # Model Configuration
    model_config:
      load_from_config:
        model_type: qwen3_dllm  # DLLM variant of Qwen3
        config:
          vocab_size: 151936
          hidden_size: 1024
          intermediate_size: 4096
          num_hidden_layers: 24
          use_cache: false

    # Dataset Configuration
    dataset_config:
      dataset_type: fineweb_edu
      dataset_format: hf_dataset
      dataset_path: HuggingFaceFW/fineweb-edu
      packing_length: 2048
      extra_kwargs:
        collator_type: dllm  # DLLM-specific data collator

    # Muon Optimizer Settings
    use_muon: true              # Enable Muon optimizer
    adam_beta1: 0.9
    adam_beta2: 0.999
    adam_epsilon: 1.0e-8
    learning_rate: 0.001
    weight_decay: 0.01

    # Training Configuration
    per_device_train_batch_size: 32
    gradient_accumulation_steps: 16
    max_steps: 10000
    warmup_steps: 1000

    # Distributed Training
    fsdp2: true  # FSDP2 for efficient distributed training
    accelerator_config:
      split_batches: true
      # If true, rank 0 loads the dataset once, splits it into `world_size` shards,
      # and dispatches each shard to the corresponding rank.
      # Ensure `per_device_train_batch_size` is divisible by `world_size`.
      # When `split_batches = true`, the effective batch per device is:
      #   per_device_train_batch_size / world_size

```

### Running Training

Use the provided script to launch multi-GPU training. 

```bash
#!/usr/bin/env bash

# Configure GPUs
GPUS=0,1,2,3
export WANDB_PROJECT="your-project-name"
export HF_HUB_DOWNLOAD_TIMEOUT=200
export HF_HUB_ETAG_TIMEOUT=200
WORLD_SIZE=$(awk -F',' '{print NF}' <<<"$GPUS")

# Launch training
CUDA_VISIBLE_DEVICES=$GPUS accelerate launch \
  --multi_gpu \
  --num_processes $WORLD_SIZE \
  --num_machines 1 \
  --machine_rank 0 \
  --main_process_ip 127.0.0.1 \
  --main_process_port 29504 \
  --dynamo_backend=no \
  --mixed_precision=no \
  --module lmms_engine.launch.cli \
  --config /path/to/your/config.yaml \
  2>&1 | tee outputs/output.log
```

For single GPU:

```bash
#!/usr/bin/env bash

GPUS=0
export WANDB_PROJECT="your-project-name"
export HF_HUB_DOWNLOAD_TIMEOUT=200
export HF_HUB_ETAG_TIMEOUT=200

# For single GPU, simply run with python (no distributed launcher needed)
CUDA_VISIBLE_DEVICES=$GPUS python -m lmms_engine.launch.cli \
  --config /path/to/your/config.yaml \
  2>&1 | tee outputs/output_single_gpu.log

```

## Key Features

### DLLM-Specific Components

1. **DLLM Trainer**: Specialized trainer (`dllm_trainer`) optimized for differential language modeling. See the code in ```src/lmms_engine/train/hf/dllm_trainer.py```.
2. **DLLM Collator**: Custom data collator (`collator_type: dllm`) for preparing batches. See example in ```src/lmms_engine/datasets/collator/text_dllm_collator.py```
4. **DLLM Model Architecture**: At its core, a DLLM behaves as a non-causal, mask-based language model. Thus, to repurpose a conventional AR model for DLLM training, simply change the attention mask from causal to full. See example in ```src/lmms_engine/models/qwen3_dllm/modeling_qwen3_dllm.py```

### Distributed Training

- **FSDP2** Recommended.
- **Deepspeed** Currently Not Support for Muon Optimizer. But It is OK for Adam Optimizer. The default config of deepspeed is provided here ```examples/ds_config/default_config.json```, where we set default zero stage as 2, but you can change it to any stage as you like.

## References

- [Muon Optimizer](https://kellerjordan.github.io/posts/muon/)
- [Diffusion Languge Models](https://www.lmms-lab.com/notes/dllm/)
