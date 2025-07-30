
# Train

To run the training, you will need to prepare a config. For data preparation, please refer to the data prep to prepare your yaml file.

Following is an example config:

```json
[
    {
        "type" : "trainer",
        "config" : {
            "trainer_type": "hf_trainer",
            "dataset_config": {
                "dataset_type" : "vision_audio",
                "dataset_format" : "yaml",
                "dataset_path" : "./scripts/yaml_files/<your_yaml>.yaml",
                "processor_config": {
                    "processor_name": "Evo-LMM/kino_qwen2_5_vl_init",
                    "processor_modality": "vision_audio",
                    "processor_type": "kino_qwen2_5"
                }
            },
            "model_config": {
                "model_name_or_path" : "Evo-LMM/kino_qwen2_5_vl_init",
                "model_class" : "qwen2_5_vl_audio",
                "attn_implementation" : "flash_attention_2"
            },
            "per_device_train_batch_size": 8,
            "learning_rate": 1e-06,
            "weight_decay": 0.0,
            "gradient_accumulation_steps": 1,
            "gradient_checkpointing": true,
            "num_train_epochs": 1,
            "save_steps": 4000,
            "save_total_limit" : 1,
            "report_to": "none",
            "output_dir": "./output/kino-7b-qwen2_5_caps_conv",
            "warmup_ratio": 0.03,
            "run_name": "kino-7b-qwen2_5_caps_conv",
            "logging_steps" : 1,
            "group_by_length" : true,
            "dataloader_num_workers" : 8,
            "bf16" : true,
            "lr_scheduler_type" : "cosine",
            "freeze_modules" : ["visual"],
            "deepspeed" : "scripts/vision/zero3.json",
            "use_liger_kernel": true
        }
    }
]
```

Let's break it down step by step:
1. The whole config will add a trainer component into the pipeline
2. For the trainer, we are building a hf trainer
3. The dataset config will be used to construct a dataset. In the dataset config, we can see that we are building a vision_audio dataset type
4. The dataset will use a `kino_qwen2_5` processor to collate the data and the hf_processor is the "Evo-LMM/kino_qwen2_5_vl_init"
5. Then for the model, we are using a `qwen2_5_vl_audio` model and using flash_attention_2
6. For the rest of the config, they are almost all parameters in the TrainingArguments where you can refer to [here](https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments)
7. For the `freeze_modules` it will freeze the component inside the model. Basically it get `model.visual` and set every parameters in it without grad. If you wish to freeze more modules, you will need to add it in the list

## Run

An examples to start the training

```bash
export NCCL_BLOCKING_WAIT=0
export TOKENIZERS_PARALLELISM=false

# CONFIG Huggingface
export HF_TOKEN="<YOUR HF_TOKEN>"
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_ENABLE_HF_TRANSFER="1"

export NCCL_DEBUG=INFO

GPUS="0,1,2,3,4,5,6,7"

CONFIG=$1

# --num_processes="${ARNOLD_WORKER_GPU}" \

torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="8000" \
    -m lmms_engine.launch.cli --config ${CONFIG}
```

