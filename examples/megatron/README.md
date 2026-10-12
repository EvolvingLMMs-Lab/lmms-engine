# Megatron multimodal SFT

This backend uses Megatron Bridge **v0.6.2** and its matching Megatron-Core commit
`a0f793dfa4e776d99a8aa63bed74e0fafc91b0db`. Both dense Qwen3-VL and Qwen3.5 have
upstream bridges, providers and Megatron models. LMMS supplies the runner and data
adapter; it does not duplicate those model implementations.

## Installation

Use Linux with Python 3.12, `uv`, Git, a C++ compiler and an NVIDIA GPU driver.
The default installer uses the repository's existing `.venv` and installs CUDA
PyTorch, Transformer Engine and Apex before Megatron. If `.venv` is absent,
it creates it with Python 3.12:

```bash
bash tools/install_megatron.sh
```

It installs PyTorch 2.13.0 / torchvision 0.28.0 with CUDA 13.0, matching CUDA
compiler packages inside the environment, TE 2.18.0 with its PyTorch extension,
and NVIDIA Apex from a pinned commit with C++/CUDA extensions enabled. TE's
PyTorch extension and Apex compile locally; the script checks the fused gradient
extension used by Megatron. The GPU driver must support the CUDA runtime.

For an NGC container that already contains CUDA PyTorch, TE and compiled Apex,
explicitly select reuse mode:

```bash
MEGATRON_RUNTIME_MODE=reuse bash tools/install_megatron.sh
```

Reuse mode requires `.venv` to already import the container's compiled libraries;
it checks them without changing the virtual environment's isolation settings.
Set `UV` if `uv` is not on PATH; `~/.local/bin/uv` is detected automatically.
`PYTHON` selects Python only when creating a missing `.venv`.
`MEGATRON_INSTALL_ROOT` selects the source and installation metadata directory
(default: `output/megatron`), while Python packages always go into `.venv`.
Existing clean source checkouts at the pinned revisions can be reused.
`MEGATRON_BRIDGE_SOURCE` and `MEGATRON_CORE_SOURCE` can point to local clones.
`MAX_JOBS` limits compilation parallelism (default: 8). By default, Apex targets
the visible GPU architectures; `TORCH_CUDA_ARCH_LIST` overrides this selection.

Qwen3.5 uses FLA 0.5.1. On Hopper with Triton >= 3.4, its gated backward operation
requires TileLang to avoid an upstream correctness issue; the installer includes
TileLang. The installer checks imports; actual hardware compatibility requires
a training run.

## Multimodal example data

The data format stays consistent with LMMS Engine's existing multimodal format,
including the `messages` schema and YAML `datasets` entries. See
[OpenMMReasoner SFT](https://huggingface.co/datasets/OpenMMReasoner/OpenMMReasoner-SFT-874K)
for an example. The example YAML files use `data/samples.parquet` and
`data/images`; set these paths to your local data.

## Launch

```bash
export CHECKPOINT_ROOT=/path/to/checkpoints
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=4
export FLA_TILELANG=1
export TORCHINDUCTOR_COMPILE_THREADS=1
unset NVTE_FLASH_ATTN NVTE_FUSED_ATTN NVTE_UNFUSED_ATTN

.venv/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=8 -m lmms_engine.launch.cli \
  --config-path "$PWD/examples/megatron" \
  --config-name qwen3_vl
```

Run from the repository root. Use `--config-name qwen3_5` for Qwen3.5-9B;
Hydra config names omit the `.yaml` extension. Append overrides such as
`trainer_args.max_steps=20 trainer_args.save_steps=20` to change settings.
The examples configure 200 optimizer steps, TP=2, PP=2, CP=2 and sequence
parallelism. Set the pack limit with `megatron_config.seq_length` and packs per
step with `trainer_args.gradient_accumulation_steps`.
`trainer_type: megatron` routes before
HF `TrainingArguments` or LMMS process-group initialization. Existing FSDP
configuration continues to select the original runner.

The runner defaults to synchronous Inductor compilation
(`TORCHINDUCTOR_COMPILE_THREADS=1`) to avoid forked compiler worker deadlocks
when Transformer Engine compiles CP kernels for new packed input shapes.
Explicitly setting this environment variable overrides the default.

The examples select `megatron_config.attention_backend=flash`, which requires
FlashAttention installed in `.venv`. The runner sets TE backend flags before
vision model construction and passes the matching backend to the language
provider. Select the backend through this Hydra setting; conflicting `NVTE_*_ATTN`
environment variables cause an error. `auto` allows TE to choose, including
cuDNN fused attention, which can require excessive backward workspace for
Qwen3-VL's vision head dimension of 72 in large packs.

A temporary runtime patch skips the bundled experimental `flash_attn.cute`
only when `flash-attn-4` is absent and its import fails because CUTLASS lacks
`ThrMma`. It runs before Bridge imports, preserves FA2 and changes no installed
package files. Compatible CuTe imports and explicitly installed FA4 are untouched.

## Performance metrics and W&B

Both examples enable W&B in **offline** mode. No login or network upload is
required. Run files are saved under `<trainer_args.output_dir>/wandb/wandb/`;
TensorBoard logs remain under `<trainer_args.output_dir>/tensorboard/`.
Only the last global rank writes W&B and TensorBoard logs. Change `logger_config.wandb_project` and
`logger_config.wandb_exp_name` with Hydra overrides, or set
`logger_config.wandb_mode=disabled` to disable W&B.

Alongside Bridge's loss, learning rate, gradient norm and memory metrics, each
`trainer_args.logging_steps` interval records:

| Metric | Meaning |
| --- | --- |
| `train/total_tokens` | Numeric cumulative unpadded decoder token count, summed across DP only and saved in the data iterator checkpoint. |
| `train/total_tokens_billions` | Numeric cumulative token count divided by 1e9 (B tokens). |
| `train/min_tokens`, `train/avg_tokens`, `train/max_tokens` | Minimum, mean and maximum unpadded token count per DP rank's microbatch over the logging interval, matching LMMS's FSDP batch-level statistics. Each packed microbatch counts as one observation, before TP/PP/CP sharding. |
| `throughput/tokens_per_sec` | Actual unpadded decoder tokens per second across the job, including visual placeholder tokens. |
| `throughput/device/tokens_per_sec` | Global token throughput divided by the total GPU count. |
| `throughput/supervised_tokens_per_sec` | Tokens with an active next-token loss per second. |
| `throughput/device/supervised_tokens_per_sec` | Supervised token throughput divided by the total GPU count. |
| `throughput/samples_per_sec` | Original conversations per second, rather than packs. |
| `throughput/step_time_seconds` | Mean optimizer-step duration over the logging interval. |
| `packing/token_utilization` | Actual tokens divided by tokens including alignment padding. |
| `throughput/estimated_tflops/device` | Estimated model training TFLOP/s per GPU. |
| `throughput/estimated_mfu_percent` | Estimated model TFLOP/s per GPU divided by its dense BF16 peak, multiplied by 100. |

Counts sum across DP only, so TP/PP/CP replicas do not multiply token throughput.
Console messages include these metrics and reuse LMMS's K/M/B token formatting;
W&B and TensorBoard receive numeric values, including both cumulative token metrics.
Checkpoints created before cumulative counting was added start the counter at zero.
Timing includes data processing, forward/backward, communication and optimizer
updates; checkpoint saving and logging are outside the measured step. Warmup
steps are included, so compare throughput after compilation has settled.

The FLOPs estimate reuses Bridge's language-model formulas, including Qwen3.5's
Gated DeltaNet layers, and adds vision FLOPs separately per image/video frame.
Attention uses separate packed sequence lengths. MFU is approximate: vision
window attention, deepstack projectors and freezing are not fully represented
by the upstream formula. Activation recomputation is excluded from model FLOPs.
Bridge's own TFLOP/s logs use its separate estimator and can differ.
`megatron_config.gpu_peak_tflops` sets the **dense, non-sparse BF16** peak per
GPU. An unset value selects 148 TFLOP/s for NVIDIA H20; on other devices, MFU is
omitted until a peak is supplied. GPU busy percentage from `nvidia-smi` is a
different metric.

## Data and memory behavior

- The existing `Qwen3_VLDataProcessor` produces image tensors, token IDs and SFT
  labels for both checkpoint families. The adapter shifts these labels once for
  Megatron's next-token loss; it preserves the processor's supervision policy.
- Each microbatch is one THD stream. Online packing groups processed samples up
  to `seq_length`, using their actual token counts. Separate cumulative lengths
  isolate attention and recurrent state across samples. The model computes MRoPE
  separately for each sample using the visual grids and packed boundaries.
- Remove padding is intrinsic to the THD representation. Small per-sample
  alignment gaps remain where TP/SP/CP require them; these gaps have zero loss.
  With the FlashAttention backend, alignment tokens are included at the end of
  each sample's attention sequence because FA2 does not support ragged gaps.
  Causal attention keeps real tokens unaffected and samples remain isolated;
  token-length metrics exclude these alignment tokens.
  There is no dense quadratic attention mask. `online_packing: false` sends one
  unpadded sample per microbatch through the same THD path.
- `per_device_train_batch_size` must be 1. Increase
  `gradient_accumulation_steps` for more packs per optimizer step. Global batch
  size counts packs, not original conversations. DP size is `WORLD_SIZE/(TP*PP*CP)`.
- BF16, FP32 gradient reduction, distributed Adam, fused gradient accumulation,
  fused vocabulary-parallel cross entropy, sequence parallelism and activation
  recomputation reduce memory use. Vision encoder and projector freezing are
  optional. Full recomputation is the default; CUDA graphs and MTP are disabled.
- Samples longer than `seq_length` are skipped as complete samples, never
  truncated through an image token span. Dataset/processor failures raise errors.
  This adapter uses deterministic sequential online packing, with preprocessing
  on each model-parallel rank; it is not a tuned high-throughput data loader.

## Checkpoints

The first run loads HF weights through the upstream bridge. Checkpoints contain
Megatron model shards, optimizer/RNG state and the data iterator's position.
Set `trainer_args.resume_from_checkpoint` to a Megatron checkpoint directory to
resume. Keep dataset order/content, processor settings, DP size and packing
settings identical. `max_steps` is the total target step, including restored
steps. The configured LR schedule is reapplied when extending a run.

Only the supported Megatron settings in the examples and `MegatronConfig` are
mapped. This runner does not implement HF Trainer callbacks, evaluation, LoRA,
FSDP options or arbitrary HF modeling execution. A new architecture needs an
upstream-compatible Megatron model/provider and a registered bridge; the common
data adapter can be reused when its input and multimodal position contracts match.
