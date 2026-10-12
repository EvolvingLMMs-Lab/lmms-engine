import time

import torch
import torch.distributed as dist
from loguru import logger
from megatron.bridge.training.callbacks import Callback
from megatron.bridge.training.utils.flop_utils import num_floating_point_operations, vit_flops
from megatron.bridge.training.utils.pg_utils import get_pg_collection

from lmms_engine.utils.train_utils import TrainUtilities


class TrainingMetrics(Callback):
    def __init__(self, gpu_peak_tflops=None):
        self.gpu_peak_tflops = gpu_peak_tflops
        self.pending = [0.0] * 6
        self.min_tokens = float("inf")
        self.max_tokens = 0
        self.elapsed = 0.0
        self.steps = 0
        self.total_tokens = 0

    def on_train_start(self, context):
        if self.gpu_peak_tflops is None and torch.cuda.get_device_name() == "NVIDIA H20":
            self.gpu_peak_tflops = 148.0
        if dist.get_rank() == 0:
            if self.gpu_peak_tflops is None:
                logger.info("MFU disabled: set megatron_config.gpu_peak_tflops to the dense BF16 peak per GPU")
            else:
                logger.info(f"Estimated MFU uses a dense BF16 peak of {self.gpu_peak_tflops:g} TFLOP/s per GPU")

    def wrap_forward_step(self, forward_step):
        def measured_forward_step(state, data_iterator, model, return_schedule_plan=False):
            def measured_iterator():
                batch = next(data_iterator)
                self.total_tokens = batch["total_consumed_tokens"]
                attention_lengths = batch["cu_seqlens_q"].diff().long()
                lengths = batch.get("sample_lengths", attention_lengths)
                batch_tokens = int(lengths.sum())
                self.min_tokens = min(self.min_tokens, batch_tokens)
                self.max_tokens = max(self.max_tokens, batch_tokens)
                padded_tokens = batch["input_ids"].numel()
                flops = num_floating_point_operations(
                    state.cfg,
                    batch_size=1,
                    seqlen_sum=padded_tokens,
                    seqlen_squared_sum=int(attention_lengths.square().sum()),
                )
                visual = batch["visual_inputs"]
                for grid in (visual.image_grid_thw, visual.video_grid_thw):
                    if grid is not None:
                        for frames, height, width in grid.tolist():
                            flops += frames * vit_flops(state.cfg, batch_size=1, num_patches=height * width)
                self.pending = [
                    total + value
                    for total, value in zip(
                        self.pending,
                        (batch_tokens, int(batch["loss_mask"].sum()), lengths.numel(), padded_tokens, flops, 1),
                    )
                ]
                yield batch

            return forward_step(state, iter(measured_iterator()), model, return_schedule_plan=return_schedule_plan)

        return measured_forward_step

    def on_train_step_start(self, context):
        torch.cuda.synchronize()
        self.started = time.perf_counter()

    def on_train_step_end(self, context):
        torch.cuda.synchronize()
        self.elapsed += time.perf_counter() - self.started
        self.steps += 1
        # Bridge increments train_state.step after this callback.
        step = context.state.train_state.step + 1
        if step % context.state.cfg.logger.log_interval == 0 or step == context.state.cfg.train.train_iters:
            self.emit(context, step)

    def emit(self, context, step):
        device = torch.cuda.current_device()
        totals = torch.tensor([*self.pending, self.total_tokens], dtype=torch.float64, device=device)
        dp_group = get_pg_collection(context.model).dp
        dist.all_reduce(totals, group=dp_group)
        min_tokens = torch.tensor(self.min_tokens, dtype=torch.float64, device=device)
        max_tokens = torch.tensor(self.max_tokens, dtype=torch.float64, device=device)
        dist.all_reduce(min_tokens, op=dist.ReduceOp.MIN, group=dp_group)
        dist.all_reduce(max_tokens, op=dist.ReduceOp.MAX, group=dp_group)
        elapsed = torch.tensor(self.elapsed, dtype=torch.float64, device=device)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        tokens, supervised, samples, padded, flops, microbatches, total_tokens = totals.tolist()
        duration = elapsed.item()
        world_size = dist.get_world_size()
        metrics = {
            "train/total_tokens": int(total_tokens),
            "train/total_tokens_billions": total_tokens / 1e9,
            "train/min_tokens": min_tokens.item(),
            "train/avg_tokens": tokens / microbatches,
            "train/max_tokens": max_tokens.item(),
            "throughput/tokens_per_sec": tokens / duration,
            "throughput/device/tokens_per_sec": tokens / duration / world_size,
            "throughput/supervised_tokens_per_sec": supervised / duration,
            "throughput/device/supervised_tokens_per_sec": supervised / duration / world_size,
            "throughput/samples_per_sec": samples / duration,
            "throughput/estimated_tflops/device": flops / duration / world_size / 1e12,
            "throughput/estimated_tflops": flops / duration / 1e12,
            "throughput/step_time_seconds": duration / self.steps,
            "packing/token_utilization": tokens / padded,
        }
        if self.gpu_peak_tflops is not None:
            metrics["throughput/estimated_mfu_percent"] = (
                100 * metrics["throughput/estimated_tflops/device"] / self.gpu_peak_tflops
            )
        if dist.get_rank() == world_size - 1:
            state = context.state
            if state.tensorboard_logger:
                for name, value in metrics.items():
                    state.tensorboard_logger.add_scalar(name, value, step)
            if state.wandb_logger:
                state.wandb_logger.log(metrics, step=step)
        if dist.get_rank() == 0:
            message = (
                f"step {step} | total tokens: {TrainUtilities.format_tokens(total_tokens)} "
                f"({metrics['train/total_tokens_billions']:.9f} B)"
            )
            for name, value in metrics.items():
                if name not in {"train/total_tokens", "train/total_tokens_billions"}:
                    message += f" | {name}: {value:.6g}"
            logger.info(message)
        self.pending = [0.0] * 6
        self.min_tokens = float("inf")
        self.max_tokens = 0
        self.elapsed = 0.0
        self.steps = 0
