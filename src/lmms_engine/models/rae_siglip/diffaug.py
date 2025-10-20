# Adapted from https://github.com/autonomousvision/stylegan-t/blob/36ab80ce76237fefe03e65e9b3161c040ae888e3/training/diffaug.py
import math

import torch
import torch.nn.functional as F


class DiffAug:
    def __init__(self, prob: float = 1.0, cutout: float = 0.2):
        self.grids = {}
        self.prob = abs(prob)
        self.using_cutout = prob > 0
        self.cutout = cutout
        self.img_channels = -1
        self.last_blur_radius = -1
        self.last_blur_kernel_h = None
        self.last_blur_kernel_w = None

    def __str__(self) -> str:
        return (
            f"DiffAug(p={self.prob}, cutout={self.cutout if self.using_cutout else 0})"
        )

    def get_grids(self, batch: int, height: int, width: int, device):
        key = (batch, height, width)
        if key in self.grids:
            return self.grids[key]
        self.grids[key] = torch.meshgrid(
            torch.arange(batch, dtype=torch.long, device=device),
            torch.arange(height, dtype=torch.long, device=device),
            torch.arange(width, dtype=torch.long, device=device),
            indexing="ij",
        )
        return self.grids[key]

    def aug(
        self, tensor: torch.Tensor, warmup_blur_schedule: float = 0.0
    ) -> torch.Tensor:
        input_dtype = tensor.dtype
        if tensor.dtype != torch.float32:
            tensor = tensor.float()

        if warmup_blur_schedule > 0:
            self.img_channels = tensor.shape[1]
            sigma0 = (tensor.shape[-2] * 0.5) ** 0.5
            sigma = sigma0 * warmup_blur_schedule
            blur_radius = math.floor(sigma * 3)
            if blur_radius >= 1:
                if self.last_blur_radius != blur_radius:
                    self.last_blur_radius = blur_radius
                    gaussian = torch.arange(
                        -blur_radius,
                        blur_radius + 1,
                        dtype=torch.float32,
                        device=tensor.device,
                    )
                    gaussian = gaussian.mul_(1 / sigma).square_().neg_().exp2_()
                    gaussian.div_(gaussian.sum())
                    kernel_h = gaussian.view(1, 1, 2 * blur_radius + 1, 1)
                    kernel_w = gaussian.view(1, 1, 1, 2 * blur_radius + 1)
                    self.last_blur_kernel_h = kernel_h.repeat(
                        self.img_channels, 1, 1, 1
                    ).contiguous()
                    self.last_blur_kernel_w = kernel_w.repeat(
                        self.img_channels, 1, 1, 1
                    ).contiguous()
                tensor = F.pad(
                    tensor,
                    [blur_radius, blur_radius, blur_radius, blur_radius],
                    mode="reflect",
                )
                tensor = F.conv2d(
                    tensor,
                    weight=self.last_blur_kernel_h,
                    bias=None,
                    groups=self.img_channels,
                )
                tensor = F.conv2d(
                    tensor,
                    weight=self.last_blur_kernel_w,
                    bias=None,
                    groups=self.img_channels,
                )

        if self.prob < 1e-6:
            return tensor

        apply_trans, apply_color, apply_cut = (torch.rand(3) <= self.prob).tolist()
        batch, device = tensor.shape[0], tensor.device
        rand_vals = (
            torch.rand(7, batch, 1, 1, device=device)
            if (apply_trans or apply_color or apply_cut)
            else None
        )

        height, width = tensor.shape[-2:]
        if apply_trans:
            ratio = 0.125
            delta_h = round(height * ratio)
            delta_w = round(width * ratio)
            translation_h = (
                rand_vals[0].mul(delta_h + delta_h + 1).floor().long() - delta_h
            )
            translation_w = (
                rand_vals[1].mul(delta_w + delta_w + 1).floor().long() - delta_w
            )

            grid_b, grid_h, grid_w = self.get_grids(batch, height, width, device)
            grid_h = (grid_h + translation_h).add_(1).clamp_(0, height + 1)
            grid_w = (grid_w + translation_w).add_(1).clamp_(0, width + 1)
            padded = F.pad(tensor, [1, 1, 1, 1, 0, 0, 0, 0])
            tensor = (
                padded.permute(0, 2, 3, 1)
                .contiguous()[grid_b, grid_h, grid_w]
                .permute(0, 3, 1, 2)
                .contiguous()
            )

        if apply_color:
            tensor = tensor.add(rand_vals[2].unsqueeze(-1).sub(0.5))
            mean = tensor.mean(dim=1, keepdim=True)
            tensor = tensor.sub(mean).mul(rand_vals[3].unsqueeze(-1).mul(2)).add_(mean)
            mean = tensor.mean(dim=(1, 2, 3), keepdim=True)
            tensor = (
                tensor.sub(mean).mul(rand_vals[4].unsqueeze(-1).add(0.5)).add_(mean)
            )

        if self.using_cutout and apply_cut:
            ratio = self.cutout
            cutout_h = round(height * ratio)
            cutout_w = round(width * ratio)
            offset_h = rand_vals[5].mul(height + (1 - cutout_h % 2)).floor().long()
            offset_w = rand_vals[6].mul(width + (1 - cutout_w % 2)).floor().long()

            grid_b, grid_h, grid_w = self.get_grids(batch, cutout_h, cutout_w, device)
            grid_h = (
                (grid_h + offset_h).sub_(cutout_h // 2).clamp_(min=0, max=height - 1)
            )
            grid_w = (
                (grid_w + offset_w).sub_(cutout_w // 2).clamp_(min=0, max=width - 1)
            )
            mask = torch.ones(batch, height, width, dtype=tensor.dtype, device=device)
            mask[grid_b, grid_h, grid_w] = 0
            tensor = tensor.mul(mask.unsqueeze(1))

        # Convert back to original dtype to preserve autocast behavior
        if input_dtype != torch.float32:
            tensor = tensor.to(input_dtype)

        return tensor
