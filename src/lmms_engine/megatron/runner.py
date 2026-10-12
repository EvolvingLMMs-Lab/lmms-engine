import os
from pathlib import Path

from .config import MegatronConfig


class MegatronRunner:
    def __init__(self, config):
        self.config = config
        self.options = MegatronConfig(**config.get("megatron_config", {}))
        self.options.validate()

    def build(self):
        os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
        import torch
        import torch._inductor.config as inductor_config

        # Avoid forked compiler worker deadlocks when TE compiles CP kernels.
        inductor_config.compile_threads = int(os.environ["TORCHINDUCTOR_COMPILE_THREADS"])
        backend_flags = {
            "auto": (1, 1, 1),
            "flash": (1, 0, 0),
            "fused": (0, 1, 0),
            "unfused": (0, 0, 1),
            "local": (0, 0, 0),
        }
        for name, enabled in zip(
            ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN"),
            backend_flags[self.options.attention_backend],
        ):
            current = os.environ.get(name)
            if current is not None and current != str(enabled):
                raise ValueError(
                    f"{name}={current} conflicts with megatron_config.attention_backend={self.options.attention_backend}; "
                    "unset NVTE_FLASH_ATTN, NVTE_FUSED_ATTN and NVTE_UNFUSED_ATTN and select the backend in the config"
                )
            os.environ[name] = str(enabled)
        from .patches import patch_bundled_flash_attention_cute

        patch_bundled_flash_attention_cute()
        from megatron.bridge import AutoBridge
        from megatron.core.transformer.enums import AttnBackend
        from megatron.bridge.training.config import (
            CheckpointConfig,
            ConfigContainer,
            DistributedDataParallelConfig,
            LoggerConfig,
            OptimizerConfig,
            RNGConfig,
            SchedulerConfig,
            TokenizerConfig,
            TrainingConfig,
            ValidationConfig,
        )
        from megatron.bridge.training.utils.checkpoint_utils import checkpoint_exists
        from transformers import AutoConfig

        from .adapters.dataset import LMMSDatasetProvider

        args, options = self.config["trainer_args"], self.options
        if args.get("max_steps", -1) < 1:
            raise ValueError("Megatron training requires trainer_args.max_steps > 0")
        if args.get("per_device_train_batch_size", 1) != 1:
            raise ValueError("THD batches use per_device_train_batch_size=1; increase gradient_accumulation_steps")
        if args.get("gradient_accumulation_steps", 1) < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        dataset_config = self.config["dataset_config"]
        if dataset_config.get("dataset_type") != "qwen3_vl":
            raise ValueError("The Megatron adapter requires dataset_type=qwen3_vl")
        if dataset_config["processor_config"]["processor_type"] != "qwen3_vl":
            raise ValueError("The Megatron adapter requires processor_type=qwen3_vl")
        resume = args.get("resume_from_checkpoint")
        if resume and (not isinstance(resume, str) or not checkpoint_exists(resume)):
            raise ValueError("resume_from_checkpoint must point to an existing Megatron checkpoint")
        model_path = self.config["model_config"]["load_from_pretrained_path"]
        hf_config = AutoConfig.from_pretrained(model_path)
        if hf_config.model_type not in {"qwen3_vl", "qwen3_5"}:
            raise ValueError("The Megatron VLM adapter currently supports dense Qwen3-VL and Qwen3.5")
        provider = AutoBridge.from_hf_config(hf_config).to_megatron_provider(load_weights=False)
        provider.attention_backend = AttnBackend[options.attention_backend]
        for name in (
            "tensor_model_parallel_size",
            "pipeline_model_parallel_size",
            "context_parallel_size",
            "sequence_parallel",
            "seq_length",
            "freeze_vision_model",
            "freeze_vision_projection",
        ):
            setattr(provider, name, getattr(options, name))
        provider.pipeline_dtype = torch.bfloat16
        provider.variable_seq_lengths = options.pipeline_model_parallel_size > 1
        provider.moe_token_dispatcher_type = "alltoall"
        provider.overlap_p2p_comm = False
        provider.mtp_num_layers = 0
        provider.calculate_per_token_loss = True
        provider.gradient_accumulation_fusion = True
        provider.cross_entropy_loss_fusion = True
        provider.cross_entropy_fusion_impl = "native"
        provider.recompute_granularity = options.recompute_granularity
        if options.recompute_granularity == "full":
            provider.recompute_method = options.recompute_method
            provider.recompute_num_layers = options.recompute_num_layers
        dp = int(os.environ.get("WORLD_SIZE", "1")) // options.model_parallel_size
        output = Path(args["output_dir"])
        logging = self.config.get("logger_config", {})
        wandb_mode = logging.get("wandb_mode", "offline")
        if wandb_mode not in {"offline", "online", "disabled"}:
            raise ValueError("logger_config.wandb_mode must be offline, online, or disabled")
        os.environ["WANDB_MODE"] = wandb_mode
        wandb_dir = output / "wandb"
        if logging.get("wandb_project"):
            wandb_dir.mkdir(parents=True, exist_ok=True)
        lr, wd = args.get("learning_rate", 5e-6), args.get("weight_decay", 0.0)
        self.bridge_config = ConfigContainer(
            model=provider,
            train=TrainingConfig(
                train_iters=args["max_steps"],
                micro_batch_size=1,
                global_batch_size=dp * args.get("gradient_accumulation_steps", 1),
            ),
            optimizer=OptimizerConfig(
                optimizer="adam",
                lr=lr,
                min_lr=0.0,
                weight_decay=wd,
                bf16=True,
                adam_beta1=args.get("adam_beta1", 0.9),
                adam_beta2=args.get("adam_beta2", 0.95),
                adam_eps=args.get("adam_epsilon", 1e-8),
                clip_grad=args.get("max_grad_norm", 1.0),
                use_distributed_optimizer=True,
            ),
            scheduler=SchedulerConfig(
                lr_decay_style=args.get("lr_scheduler_type", "cosine"),
                lr_decay_iters=args["max_steps"],
                lr_warmup_iters=args.get("warmup_steps", 0),
                start_weight_decay=wd,
                end_weight_decay=wd,
                weight_decay_incr_style="constant",
                override_opt_param_scheduler=True,
            ),
            ddp=DistributedDataParallelConfig(
                use_distributed_optimizer=True,
                grad_reduce_in_fp32=True,
                check_for_nan_in_grad=True,
                average_in_collective=False,
                overlap_grad_reduce=False,
                overlap_param_gather=False,
            ),
            dataset=LMMSDatasetProvider(
                lmms_dataset=self.config["dataset_config"],
                seq_length=options.seq_length,
                alignment=options.alignment,
                online_packing=options.online_packing,
                attention_backend=options.attention_backend,
            ),
            validation=ValidationConfig(eval_iters=0, eval_interval=args["max_steps"] + 1),
            logger=LoggerConfig(
                log_interval=args.get("logging_steps", 1),
                tensorboard_log_interval=args.get("logging_steps", 1),
                tensorboard_dir=str(output / "tensorboard"),
                log_memory_to_tensorboard=True,
                log_throughput=True,
                log_timers_to_tensorboard=True,
                wandb_project=logging.get("wandb_project"),
                wandb_exp_name=logging.get("wandb_exp_name") or output.name,
                wandb_entity=logging.get("wandb_entity"),
                wandb_save_dir=str(wandb_dir),
            ),
            tokenizer=TokenizerConfig(tokenizer_type="HuggingFaceTokenizer", tokenizer_model=model_path),
            checkpoint=CheckpointConfig(
                pretrained_checkpoint=model_path,
                save=str(output / "checkpoints"),
                load=args.get("resume_from_checkpoint"),
                save_interval=args.get("save_steps", args["max_steps"]),
                ckpt_format="torch_dist",
                fully_parallel_save=True,
            ),
            rng=RNGConfig(seed=args.get("seed", 42)),
            mixed_precision="bf16_mixed",
        )

    def run(self):
        from megatron.bridge.training.finetune import finetune
        from megatron.bridge.training.vlm_step import forward_step

        from .metrics import TrainingMetrics

        metrics = TrainingMetrics(gpu_peak_tflops=self.options.gpu_peak_tflops)
        finetune(self.bridge_config, metrics.wrap_forward_step(forward_step), callbacks=[metrics])
