# LMMs Engine

Training framework for LMMs-Lab.

## Installation

Installation is simple

```bash
uv sync
```

## Launch

The recommended way to launch is always use torchrun as it is the most native way to launch torch and in most of the settings this should work. Most of the debug and development should be based on this as we might not always use accelerate in our later framework.

```bash
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=12355 -m lmms_engine.launch.cli --config examples/load_from_pretrained_example.yaml
```

## Examples

We provide two examples here to demonstrate how to use the training engine in most of the case, you will need to perform the following three steps:

1. Process the dataset into a specific format and store it in (jsonl/json/arrow)
2. Write your dataset yaml (Optional if you are only using a single data source)
3. Prepare your training config

### 1. Process the dataset

You will need to process the dataset in OpenAI chat messages format. We prepare an example for you to reference. You can get the data by using

```bash
hf download kcz358/open-thoughts-debug --local-dir data/open_thoughts_debug --repo-type dataset
```

### 2. Prepare dataset yaml

You can specify the data by using the following yaml, data folder can be left empty for text dataset.

```yaml
datasets:
  - path: data/open_thoughts_debug
    data_folder: ""
    data_type: arrow
```

### 3. Prepare training config

The last step would be to prepare the training config. We support fsdp2 and deepspeed zero

Please check the [config_example.yaml](examples/config_example.yaml) for more details.

## More Content

- [Preparing Data and how the data is load](docs/data_prep.md)
- [Overall Design Principle](docs/design_principle.md)
- [Training](docs/train.md)
- [API](docs/api.md)


#### Current Supported Ops

- Qwen2 or 2.5 LM series
- Qwen2.5 VL
- QwenAudioEncoder

To use rmpad, you should install flash-attn also. You can do it by

```bash
uv pip install flash-attn --no-build-isolation
```

If you encounter any issue for example symbol not found. This is possibly because of the flash-attn has been compiled on the wrong torch version. You can run

```bash
uv pip install --no-build-isolation --no-cache-dir flash-attn
```

To use it, you will need to set

```yaml
use_liger_kernel: true
use_rmpad: true
```

in the training config. Then the forward would be patched into the model.

### Sequence Packing

Sequence packing is a techniques to accelerate the training process by removing the pad. With it enabled, it will boost the training performance quickly. Currently the implementation is being fused with liger-kernel and being patched to the model's forward during training. Thus, we might need to validate the operations. Current sequence packing ops are all written in flash-attn with the `var_len` function so we need to install `flash-attn` and `liger-kernel` to use it. If you currently use the fully unpadding techniques start from the input ids, the MFU can reach to about 35-40 under ideal settings. Normally, in most of the cases, a range between 25-35 would be normal

### Liger Kernel

[Liger Kernel](https://github.com/linkedin/Liger-Kernel) is a collection of Triton kernels designed specifically for LLM training. It can effectively increase multi-GPU training throughput and reduces memory usage. Based on my testing, it does reduces memory usage when finetuning models. Benchmarking based on my testing under kino stage-1 training settings, it reduces the memory usage by around 30%. The major memory reduction is on the fused CrossEntropy kernel and allow us to use large batch size during training.

To use it is simple, you need to first install it using `uv pip install liger-kernel`. Then set the `use_liger_kernel` in the trainer config to `true`. The patching logic currently is as follows:

1. For our custom model, you will need to write your own `apply_liger_kernel_to_xxx` and register the model type to the `MODEL_REGISTRY` in the monkey patch.
2. If the model is not in the registry, we will search if it is in the original liger-kernel implementation
3. If the model is not in the registry, we will see if it contains a `language_model` component and apply liger-kernel on that