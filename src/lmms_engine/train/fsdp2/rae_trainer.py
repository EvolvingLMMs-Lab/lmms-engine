import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from loguru import logger

from lmms_engine.models.rae_siglip import ProjectedDiscriminator
from lmms_engine.models.rae_siglip.diffaug import DiffAug
from lmms_engine.models.rae_siglip.lpips import LPIPS
from lmms_engine.train.registry import TRAINER_REGISTER
from lmms_engine.utils.fsdp2_utils import (
    fsdp2_clip_grad_norm_,
    get_constant_schedule,
    get_cosine_schedule_with_warmup,
    get_wsd_schedule_with_warmup,
)

from .fsdp2_trainer import FSDP2SFTTrainer


@TRAINER_REGISTER.register("rae_trainer")
class RaeTrainer(FSDP2SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Hardcoded EMA decay configuration to match original RAE Stage-1 training
        self.ema_decay = 0.9978  # From original RAE config (DINOv2-B_decXL.yaml)
        # Hardcoded discriminator configuration
        # Matches original RAE Stage-1 training config (DINOv2-B_decXL.yaml)
        self.discriminator_config = {
            "disc_weight": 0.75,
            "perceptual_weight": 1.0,
            "disc_updates": 1,
            "max_d_weight": 10000.0,
            "disc_loss": "hinge",
            "gen_loss": "vanilla",
            # Original RAE uses epoch-based schedule:
            # disc_upd_start: 6 (discriminator updates start at epoch 6)
            # disc_start: 8 (GAN loss starts at epoch 8)
            # For 16 epochs: 6/16 = 0.375, 8/16 = 0.5
            "discriminator_start": 6,  # epoch 6 - discriminator updates start
            "discriminator_loss_start": 8,  # epoch 8 - GAN loss to generator starts
            "lpips_start": 0,  # epoch 0 - LPIPS loss starts immediately
            "discriminator_lr": 2.0e-04,
            "discriminator_weight_decay": 0.0,
            "discriminator_lr_scheduler_type": "cosine",
            "discriminator_warmup_ratio": 0.0625,  # 1 epoch out of 16
            "dino_ckpt_path": "./data/discs/dino_vit_small_patch8_224.pth",
            "ks": 9,
            "key_depths": (2, 5, 8, 11),
            "norm_type": "bn",
            "using_spec_norm": True,
            "norm_eps": 1e-6,
            "recipe": "S_8",
            "augment": {
                "prob": 1.0,
                "cutout": 0.0,
            },
        }

        dino_ckpt_path = self.discriminator_config.get("dino_ckpt_path")
        if not dino_ckpt_path or not os.path.isfile(dino_ckpt_path):
            raise FileNotFoundError(
                f"DINO checkpoint not found at {dino_ckpt_path}. Please provide the pretrained discriminator weights."
            )
        self.discriminator = ProjectedDiscriminator(
            dino_ckpt_path=dino_ckpt_path,
            ks=int(self.discriminator_config.get("ks", 9)),
            key_depths=tuple(self.discriminator_config.get("key_depths", (2, 5, 8, 11))),
            norm_type=self.discriminator_config.get("norm_type", "bn"),
            using_spec_norm=bool(self.discriminator_config.get("using_spec_norm", True)),
            norm_eps=float(self.discriminator_config.get("norm_eps", 1e-6)),
            recipe=self.discriminator_config.get("recipe", "S_8"),
        )

        augment_cfg = self.discriminator_config.get("augment", {})
        prob = float(augment_cfg.get("prob", 1.0))
        cutout = float(augment_cfg.get("cutout", 0.0))
        self.diffaug = DiffAug(prob=prob, cutout=cutout) if prob > 0 else None

        self.perceptual_weight = float(self.discriminator_config.get("perceptual_weight", 1.0))
        self.disc_weight = float(self.discriminator_config.get("disc_weight", 1.0))
        self.max_d_weight = float(self.discriminator_config.get("max_d_weight", 1e4))
        self.disc_updates = int(self.discriminator_config.get("disc_updates", 1))

        self.disc_loss_type = self.discriminator_config.get("disc_loss", "hinge")
        self.gen_loss_type = self.discriminator_config.get("gen_loss", "vanilla")
        self.gen_loss_fn = self._select_gen_loss(self.gen_loss_type)
        self.disc_loss_fn = self._select_disc_loss(self.disc_loss_type)

        self.lpips = LPIPS(use_dropout=True)
        self.lpips.eval()

        self.lpips_start_epoch = int(self.discriminator_config.get("lpips_start", 0))

        # ema_decay is now set in __init__ from extra_kwargs
        self.ema_state = None
        self.last_layer_module = None
        self.lpips_start_step = 0
        self.discriminator_start_step = 0
        self.discriminator_loss_start = 0

    def prepare_model(self):
        super().prepare_model()
        device = next(self.fsdp2_model.parameters()).device
        self.discriminator.build(device)
        self.lpips.to(device)
        self.discriminator.train()
        self.discriminator.to(device=device, dtype=self.fsdp2_model.dtype)

        # Fix: Store the decoder_pred module instead of the weight parameter directly
        # This is necessary for FSDP2 compatibility
        if hasattr(self.fsdp2_model, "module"):
            self.last_layer_module = self.fsdp2_model.module.decoder.decoder_pred
        else:
            self.last_layer_module = self.fsdp2_model.decoder.decoder_pred
        logger.info(f"[RAE Trainer] Using last layer module: {type(self.last_layer_module)}")

        if self.ema_decay is not None and self.ema_decay > 0:
            self._init_ema_state()

    def _init_ema_state(self):
        self.ema_state = {}
        for name, param in self.fsdp2_model.named_parameters():
            if param.requires_grad:
                self.ema_state[name] = param.detach().clone()

    def _update_ema(self):
        if self.ema_state is None:
            return
        decay = float(self.ema_decay)
        with torch.no_grad():
            for name, param in self.fsdp2_model.named_parameters():
                if not param.requires_grad:
                    continue
                ema_param = self.ema_state[name]
                ema_param.mul_(decay).add_(param.detach(), alpha=1 - decay)

    def lpips_should_start(self):
        return self.global_step >= self.lpips_start_step

    def compute_loss(self, batch):
        if self.args.bf16:
            cast_dtype = torch.bfloat16
        else:
            cast_dtype = torch.float16
        loss_kwargs = {}
        self.discriminator.eval()
        pixel_values = batch["pixel_values"].to(next(self.fsdp2_model.parameters()).device)

        # Debug: Log pixel value ranges periodically
        if self.global_step % 100 == 0 and dist.get_rank() == 0:
            logger.info(
                f"[Step {self.global_step}] Input pixels - min: {pixel_values.min().item():.4f}, max: {pixel_values.max().item():.4f}, mean: {pixel_values.mean().item():.4f}"
            )

        # Forward pass in mixed precision
        with torch.autocast(device_type="cuda", dtype=cast_dtype):
            outputs = self.fsdp2_model(pixel_values=pixel_values)
            recon = outputs["out_pixels"]

        # Debug: Log reconstruction value ranges periodically
        if self.global_step % 100 == 0 and dist.get_rank() == 0:
            logger.info(
                f"[Step {self.global_step}] Recon pixels - min: {recon.min().item():.4f}, max: {recon.max().item():.4f}, mean: {recon.mean().item():.4f}"
            )

        # Compute losses in float32 for numerical stability
        rec_loss = F.l1_loss(recon.float(), pixel_values.float())

        if self.perceptual_weight > 0 and self.lpips_should_start():
            lpips_loss = self.lpips(recon.float(), pixel_values.float())
        else:
            lpips_loss = torch.zeros_like(rec_loss)

        recon_total = rec_loss + self.perceptual_weight * lpips_loss

        gan_loss = torch.zeros_like(recon_total)
        adaptive_weight = torch.zeros_like(recon_total)
        if self.disc_weight > 0 and self.discriminator_should_start_loss():
            # Normalize to [-1, 1] for discriminator (in float32 to avoid dtype issues)
            recon_normed = (recon * 2.0 - 1.0).float()
            fake_aug = self.diffaug.aug(recon_normed) if self.diffaug else recon_normed

            with torch.autocast(device_type="cuda", dtype=cast_dtype):
                logits_fake = self.discriminator(fake_aug)
                gan_loss = self.gen_loss_fn(logits_fake)

            adaptive_weight = self._calculate_adaptive_weight(recon_total, gan_loss)
            recon_total = recon_total + self.disc_weight * adaptive_weight * gan_loss

        # Debug: Log loss components periodically
        if self.global_step % 100 == 0 and dist.get_rank() == 0:
            logger.info(
                f"[Step {self.global_step}] Losses - L1: {rec_loss.item():.6f}, LPIPS: {lpips_loss.item():.6f}, GAN: {gan_loss.item():.6f}, AdaptiveW: {adaptive_weight.item():.6f}"
            )

        loss_kwargs["l1_loss"] = rec_loss.detach().item()
        loss_kwargs["lpips_loss"] = lpips_loss.detach().item()
        loss_kwargs["gan_loss"] = gan_loss.detach().item()
        loss_kwargs["adaptive_weight"] = adaptive_weight.detach().item()
        return recon_total, loss_kwargs

    def compute_loss_discriminator(self, batch):
        if self.args.bf16:
            cast_dtype = torch.bfloat16
        else:
            cast_dtype = torch.float16
        device = next(self.fsdp2_model.parameters()).device
        pixel_values = batch["pixel_values"].to(device)
        real_normed = (pixel_values * 2.0 - 1.0).float()
        with torch.no_grad():
            outputs = self.fsdp2_model(pixel_values=pixel_values)
            fake_pixels = outputs["out_pixels"]
        fake_normed = fake_pixels * 2.0 - 1.0
        fake_normed = fake_normed.clamp(-1.0, 1.0)
        fake_normed = (torch.round((fake_normed + 1.0) * 127.5) / 127.5 - 1.0).float()

        # Apply augmentation in float32 to avoid dtype issues
        fake_input = self.diffaug.aug(fake_normed) if self.diffaug else fake_normed
        real_input = self.diffaug.aug(real_normed) if self.diffaug else real_normed

        with torch.autocast(device_type="cuda", dtype=cast_dtype):
            logits_fake = self.discriminator(fake_input)
            logits_real = self.discriminator(real_input)
            loss_discriminator = self.disc_loss_fn(logits_real, logits_fake)
        return loss_discriminator

    def prepare_optimizer(self):
        super().prepare_optimizer()
        discriminator_lr = self.discriminator_config.get("discriminator_lr", self.args.learning_rate)
        discriminator_weight_decay = self.discriminator_config.get("discriminator_weight_decay", self.args.weight_decay)
        # Use same betas as generator (matching original RAE config: [0.5, 0.9])
        self.discriminator_optimizer = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=discriminator_lr,
            weight_decay=discriminator_weight_decay,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
        )

    def prepare_scheduler(self, num_warmup_steps: int, num_training_steps: int):
        super().prepare_scheduler(num_warmup_steps, num_training_steps)
        total_epochs = max(int(getattr(self.args, "num_train_epochs", 1)), 1)
        self.steps_per_epoch = max(num_training_steps // total_epochs, 1)

        discriminator_start_ratio = self.discriminator_config.get("discriminator_start_ratio")
        discriminator_loss_start_ratio = self.discriminator_config.get("discriminator_loss_start_ratio")

        if "discriminator_start" in self.discriminator_config:
            self.discriminator_start_step = int(self.discriminator_config["discriminator_start"] * self.steps_per_epoch)
        elif "discriminator_start_step" in self.discriminator_config:
            self.discriminator_start_step = int(self.discriminator_config["discriminator_start_step"])
        elif discriminator_start_ratio is not None:
            self.discriminator_start_step = int(num_training_steps * discriminator_start_ratio)
        else:
            self.discriminator_start_step = 0

        if "discriminator_loss_start" in self.discriminator_config:
            self.discriminator_loss_start = int(
                self.discriminator_config["discriminator_loss_start"] * self.steps_per_epoch
            )
        elif "discriminator_loss_start_step" in self.discriminator_config:
            self.discriminator_loss_start = int(self.discriminator_config["discriminator_loss_start_step"])
        elif discriminator_loss_start_ratio is not None:
            self.discriminator_loss_start = int(num_training_steps * discriminator_loss_start_ratio)
        else:
            self.discriminator_loss_start = 0

        if "lpips_start_step" in self.discriminator_config:
            self.lpips_start_step = int(self.discriminator_config["lpips_start_step"])
        else:
            self.lpips_start_step = int(self.lpips_start_epoch * self.steps_per_epoch)

        # Debug: Log training schedule info
        logger.info(f"[RAE Trainer] Training schedule:")
        logger.info(f"  - Total steps: {num_training_steps}")
        logger.info(f"  - Steps per epoch: {self.steps_per_epoch}")
        logger.info(f"  - LPIPS start step: {self.lpips_start_step}")
        logger.info(f"  - Discriminator start step: {self.discriminator_start_step}")
        logger.info(f"  - Discriminator loss start step: {self.discriminator_loss_start}")

        self.discriminator_training_steps = max(num_training_steps - self.discriminator_start_step, 1)

        discriminator_warmup_ratio = self.discriminator_config.get("discriminator_warmup_ratio", 0.1)
        self.discriminator_warmup_steps = int(self.discriminator_training_steps * discriminator_warmup_ratio)

        # Get discriminator scheduler type
        discriminator_lr_scheduler_type = self.discriminator_config.get(
            "discriminator_lr_scheduler_type", self.args.lr_scheduler_type
        )

        if discriminator_lr_scheduler_type == "cosine":
            self.discriminator_scheduler = get_cosine_schedule_with_warmup(
                self.discriminator_optimizer,
                num_warmup_steps=self.discriminator_warmup_steps,
                num_training_steps=self.discriminator_training_steps,
                **self.args.lr_scheduler_kwargs,
            )
        elif discriminator_lr_scheduler_type == "wsd":
            self.discriminator_scheduler = get_wsd_schedule_with_warmup(
                self.discriminator_optimizer,
                num_warmup_steps=self.discriminator_warmup_steps,
                num_training_steps=self.discriminator_training_steps,
                **self.args.lr_scheduler_kwargs,
            )
        elif discriminator_lr_scheduler_type == "constant":
            self.discriminator_scheduler = get_constant_schedule(
                self.discriminator_optimizer,
                num_warmup_steps=self.discriminator_warmup_steps,
                **self.args.lr_scheduler_kwargs,
            )
        else:
            raise ValueError(f"Unsupported discriminator_lr_scheduler_type: {discriminator_lr_scheduler_type}")

    def training_step(self, batch):
        # Train generator
        self.fsdp2_model.train()
        self.optimizer.zero_grad()
        loss, loss_kwargs = self.compute_loss(batch)
        if dist.get_world_size() > 1:
            loss = loss.mean()
        loss_item = loss.item()
        loss.backward()
        grad_norm = fsdp2_clip_grad_norm_(self.fsdp2_model.parameters(), self.args.max_grad_norm)

        # Debug: Log gradient norm periodically
        if self.global_step % 100 == 0 and dist.get_rank() == 0:
            logger.info(f"[Step {self.global_step}] Generator grad_norm: {grad_norm.item():.6f}")

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            logger.warning(f"[Step {self.global_step}] WARN: grad_norm is not finite: {grad_norm}")
            self.optimizer.zero_grad()
        else:
            self.optimizer.step()
            self._update_ema()

        self.scheduler.step()

        # reduce loss across dp ranks
        lr = self.scheduler.get_last_lr()[0]
        loss_item = torch.tensor(loss_item, device=self.args.device)
        torch.distributed.all_reduce(loss_item, op=torch.distributed.ReduceOp.AVG)

        # Train discriminator
        # Re-generate fake images with no_grad to avoid memory overhead
        loss_discriminator_item = 0.0
        discriminator_lr = 0.0
        if self.disc_weight > 0 and self.discriminator_should_start():
            self.discriminator.train()
            disc_loss_total = 0.0
            for _ in range(max(self.disc_updates, 1)):
                self.discriminator_optimizer.zero_grad()
                loss_discriminator = self.compute_loss_discriminator(batch)
                disc_loss_total += loss_discriminator.item()
                loss_discriminator.backward()
                self.discriminator_optimizer.step()
                if self.discriminator_scheduler is not None:
                    self.discriminator_scheduler.step()

            loss_discriminator_item = disc_loss_total / max(self.disc_updates, 1)
            if self.discriminator_scheduler is not None:
                discriminator_lr = self.discriminator_scheduler.get_last_lr()[0]
            else:
                discriminator_lr = self.discriminator_optimizer.param_groups[0]["lr"]

            loss_discriminator_tensor = torch.tensor(loss_discriminator_item, device=self.args.device)
            torch.distributed.all_reduce(loss_discriminator_tensor, op=torch.distributed.ReduceOp.AVG)
            loss_discriminator_item = loss_discriminator_tensor.item()

        for k, v in loss_kwargs.items():
            if not isinstance(v, torch.Tensor):
                loss_kwargs[k] = torch.tensor(v, device=self.args.device)
            torch.distributed.all_reduce(loss_kwargs[k], op=torch.distributed.ReduceOp.AVG)
            loss_kwargs[k] = loss_kwargs[k].item()

        result = {
            "train/loss": loss_item.item(),
            "train/lr": lr,
            "train/grad_norm": grad_norm.item(),
            "train/discriminator_loss": loss_discriminator_item,
            "train/discriminator_lr": discriminator_lr,
        }
        for k, v in loss_kwargs.items():
            result[f"train/{k}"] = v

        return result

    def discriminator_should_start(self):
        return self.global_step >= self.discriminator_start_step

    def discriminator_should_start_loss(self):
        return self.global_step >= self.discriminator_loss_start

    def _select_gen_loss(self, loss_type: str):
        if loss_type == "hinge":
            return lambda logits_fake: -logits_fake.mean()
        if loss_type == "vanilla":
            return lambda logits_fake: F.softplus(-logits_fake).mean()
        raise ValueError(f"Unsupported generator loss: {loss_type}")

    def _select_disc_loss(self, loss_type: str):
        if loss_type == "hinge":

            def hinge_loss(logits_real, logits_fake):
                loss_real = F.relu(1.0 - logits_real).mean()
                loss_fake = F.relu(1.0 + logits_fake).mean()
                return 0.5 * (loss_real + loss_fake)

            return hinge_loss
        if loss_type == "vanilla":

            def vanilla_loss(logits_real, logits_fake):
                loss_real = F.softplus(-logits_real).mean()
                loss_fake = F.softplus(logits_fake).mean()
                return loss_real + loss_fake

            return vanilla_loss
        raise ValueError(f"Unsupported discriminator loss: {loss_type}")

    def save_checkpoints(self, output_path: str, step: int, total_limit: int = None):
        """Save model checkpoint including EMA weights."""
        super().save_checkpoints(output_path, step, total_limit)

        # Save EMA state if it exists
        if self.ema_state is not None and dist.get_rank() == 0:
            ema_path = os.path.join(output_path, "ema_state.pt")
            logger.info(f"[RAE Trainer] Saving EMA state to {ema_path}")
            torch.save(self.ema_state, ema_path)
            logger.info(f"[RAE Trainer] EMA state saved successfully")

    def load_checkpoints(self, output_path: str, step: int):
        """Load model checkpoint including EMA weights."""
        super().load_checkpoints(output_path, step)

        # Load EMA state if it exists
        ema_path = os.path.join(output_path, "ema_state.pt")
        if os.path.exists(ema_path):
            logger.info(f"[RAE Trainer] Loading EMA state from {ema_path}")
            self.ema_state = torch.load(ema_path, map_location="cpu")
            logger.info(f"[RAE Trainer] EMA state loaded successfully")
        else:
            logger.warning(f"[RAE Trainer] EMA state file not found at {ema_path}")

    def _calculate_adaptive_weight(self, recon_loss: torch.Tensor, gan_loss: torch.Tensor) -> torch.Tensor:
        if gan_loss.requires_grad is False:
            if self.global_step % 100 == 0 and dist.get_rank() == 0:
                logger.warning(f"[Step {self.global_step}] GAN loss has no gradient, returning zero adaptive weight")
            return torch.zeros_like(recon_loss)

        # Get the weight parameter from the last layer module
        last_layer_weight = self.last_layer_module.weight
        if last_layer_weight is None:
            if self.global_step % 100 == 0 and dist.get_rank() == 0:
                logger.warning(f"[Step {self.global_step}] Last layer weight is None, returning zero adaptive weight")
            return torch.zeros_like(recon_loss)

        grad_recon = torch.autograd.grad(recon_loss, last_layer_weight, retain_graph=True, allow_unused=True)[0]
        grad_gan = torch.autograd.grad(gan_loss, last_layer_weight, retain_graph=True, allow_unused=True)[0]

        if grad_recon is None or grad_gan is None:
            if self.global_step % 100 == 0 and dist.get_rank() == 0:
                logger.warning(
                    f"[Step {self.global_step}] Gradients are None - grad_recon: {grad_recon is not None}, grad_gan: {grad_gan is not None}"
                )
            return torch.zeros_like(recon_loss)

        grad_recon_norm = torch.norm(grad_recon)
        grad_gan_norm = torch.norm(grad_gan)
        d_weight = grad_recon_norm / (grad_gan_norm + 1e-6)
        d_weight = torch.clamp(d_weight, 0.0, self.max_d_weight)

        # Debug: Log gradient norms and adaptive weight periodically
        if self.global_step % 100 == 0 and dist.get_rank() == 0:
            logger.info(
                f"[Step {self.global_step}] Adaptive weight calc - grad_recon_norm: {grad_recon_norm.item():.6f}, grad_gan_norm: {grad_gan_norm.item():.6f}, d_weight: {d_weight.item():.6f}"
            )

        return d_weight.detach()
