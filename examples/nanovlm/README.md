# NanoVLM Speedrun

> The most striking thing about the [modded-nanogpt](https://github.com/karpathy/modded-nanogpt) experiments is that they expose how much of deep learning is just bloat. 
> To apply this to Vision-Language Models (VLMs), you have to stop acting like a researcher and start acting like a hacker. You aren't trying to follow academic standards; you are trying to maximize the movement of bits through silicon.

We introduce **NanoVLM Speedrun**: a minimalist VLM recipe designed to strip away the bloat. We provide the bare-minimum components required to bridge the training and evaluation pipeline, enabling lightning-fast iteration and reproduction.

## The Recipe (2026H1)

- **LLM**: [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B )
- **Vision Encoder**: [`google/siglip2-so400m-patch16-naflex`](https://huggingface.co/google/siglip2-so400m-patch16-naflex )
- **Projector**: Classic [LLaVA](https://arxiv.org/abs/2310.03744)-style **2-layer MLP**
- **Training Paradigm**: A streamlined two-stage approach:
  - **Stage 1**: Projector-only alignment (tuning the projector between vision and language).
  - **Stage 2**: End-to-end instruction tuning (tuning both the projector and the LLM).

## Data Preparation

We utilize the curated [LMMs-Lab-Speedrun/Data_NanoVLM](https://huggingface.co/datasets/LMMs-Lab-Speedrun/Data_NanoVLM ) collection.

- **Stage 1**: From [liuhaotian/LLaVA-Pretrain](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain )
- **Stage 2**: From [lmms-lab/LLaVA-NeXT-Data](https://huggingface.co/datasets/lmms-lab/LLaVA-NeXT-Data) (Note: We explicitly filtered out excessively long samples to maintain training efficiency). 


### Dataset YAML Configuration

Configure your local paths in the YAML files as shown below:

#### Stage 1 YAML (Example)

```yaml
datasets:
  - path: LMMs-Lab-Speedrun/Data_NanoVLM/Stage1-LLaVA-Pretrain/Lmms_format_blip_laion_cc_sbu_558k.json
    data_folder: path/to/Stage1-LLaVA-Pretrain/Image
    data_type: json
```

#### Stage 2 YAML (Example)

```yaml
datasets:
  - path: LMMs-Lab-Speedrun/Data_NanoVLM/Stage2-LLaVA-NeXT-Data/llava_next_Lmms_format_processed.json
    data_folder: path/to/LLaVA-NeXT-Data/Images
    data_type: json
```

## Execution

### 0. Installation & Initialization

For environment setup, please refer to the [lmms-engine Quick Start](https://github.com/EvolvingLMMs-Lab/lmms-engine?tab=readme-ov-file#-quick-start).

Download and use [NanoVLM_Init](https://huggingface.co/datasets/LMMs-Lab-Speedrun/NanoVLM_Init) for Stage 1 initialization.

### 1. Stage 1: Pre-training

```bash
bash ./examples/nanovlm/stage1_nanovlm_train.sh
```

### 2. Merge Stage 1 Checkpoint

```bash
python -m lmms_engine.merger \
  --checkpoint_path ./output/nanovlm_stage1/checkpoint-2180 \
  --output_path ./output/nanovlm_stage1/checkpoint-2180-merged
```

### 3. Stage 2: Instruction Tuning

```bash
export DATASET_PATH="/path/to/stage2_llava_next.yaml"
bash ./examples/nanovlm/stage2_nanovlm_train.sh
```

### 4. Merge Stage 2 Checkpoint

```bash
python -m lmms_engine.merger \
  --checkpoint_path ./output/nanovlm_stage2/checkpoint-11540 \
  --output_path ./output/nanovlm_stage2/checkpoint-11540-merged
```

## Evaluation (lmms-eval)

```bash
git clone -b dev-v0.7 https://github.com/EvolvingLMMs-Lab/lmms-eval.git 
cd lmms-eval
```

Run evaluation (replace pretrained=... with your merged weights):

```bash
# Multi-GPU asynchronous evaluation
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m lmms_eval \
    --model nanovlm \
    --model_args pretrained=./output/nanovlm_stage2/checkpoint-11540-merged \
    --tasks mme \
    --batch_size 1
```

## Results

### Training Overhead

| Stage | Total FLOPS | Energy | CO2 Emissions | GPU Hours (8 x H100) |
|-------|-------------|--------|---------------|------------------|
| Stage 1 | 236.79 PFLOPS | 13.5221 kWh | 6.42 kg | 19.32 |
| Stage 2 | 98.23 PFLOPS | 3.1006 kWh | 1.47 kg | 4.43 |

### Benchmark Scores
| MME | MMMU_Val | MMStar | GQA | ChartQA | DocVQA | OCRBench | POPE |
|-----|----------|--------|-----|---------|--------|----------|------|
| 1204.46 (948.75/255.71) | 0.3022 | 0.3273 | 0.4184 | 0.1084 | 0.1018 | 0.165 | 0.7724 |

*Reported metrics: MME (Perception/Cognition); MMBench (EN Dev); MMMU_Val; MMStar (Average); GQA (Exact Match); ChartQA (Relaxed Overall); DocVQA (ANLS); OCRBench; POPE (Accuracy)*

## Launch Preparation Community Discussion Trails

- [2026.02.27] Initial NanoVLM recipe released.

## List of TODOs

- [x] Publish Stage 1 & Stage 2 training scripts.
- [x] Publish evaluation scripts.
- [ ] Add more benchmark results (MMMU, OCRBench, BLINK).
- [ ] Optimize the training framework.
```
