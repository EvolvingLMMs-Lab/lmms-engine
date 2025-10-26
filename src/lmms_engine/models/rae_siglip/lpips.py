"""LPIPS implementation aligned with the original RAE repository."""
from collections import namedtuple

import torch
import torch.nn as nn
from torchvision import models

from .lpips_utils import get_ckpt_path


class ScalingLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("shift", torch.tensor([-0.030, -0.088, -0.188])[None, :, None, None])
        self.register_buffer("scale", torch.tensor([0.458, 0.448, 0.450])[None, :, None, None])

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.shift) / self.scale


class NetLinLayer(nn.Module):
    def __init__(self, chn_in: int, chn_out: int = 1, use_dropout: bool = False) -> None:
        super().__init__()
        layers = [nn.Dropout()] if use_dropout else []
        layers.append(nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False))
        self.model = nn.Sequential(*layers)


class VGG16FeatureExtractor(nn.Module):
    def __init__(self, requires_grad: bool = False, pretrained: bool = True) -> None:
        super().__init__()
        features = models.vgg16(pretrained=pretrained).features
        self.slice1 = nn.Sequential(*[features[x] for x in range(4)])
        self.slice2 = nn.Sequential(*[features[x] for x in range(4, 9)])
        self.slice3 = nn.Sequential(*[features[x] for x in range(9, 16)])
        self.slice4 = nn.Sequential(*[features[x] for x in range(16, 23)])
        self.slice5 = nn.Sequential(*[features[x] for x in range(23, 30)])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, tensor: torch.Tensor):
        h = self.slice1(tensor)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        outputs = namedtuple("VggOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"])
        return outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)


def _normalize(tensor: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    norm_factor = torch.sqrt(torch.sum(tensor**2, dim=1, keepdim=True))
    return tensor / (norm_factor + eps)


def _spatial_average(tensor: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
    return tensor.mean([2, 3], keepdim=keepdim)


class LPIPS(nn.Module):
    def __init__(self, use_dropout: bool = True) -> None:
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]
        self.net = VGG16FeatureExtractor(pretrained=True, requires_grad=False)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self._load_pretrained_weights()
        for param in self.parameters():
            param.requires_grad = False

    def _load_pretrained_weights(self, name: str = "vgg_lpips") -> None:
        ckpt = get_ckpt_path(name)
        state = torch.load(ckpt, map_location=torch.device("cpu"))
        self.load_state_dict(state, strict=False)
        print(f"[LPIPS] Loaded pretrained weights from {ckpt}")

    def forward(self, input: torch.Tensor, target: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        input_scaled, target_scaled = self.scaling_layer(input), self.scaling_layer(target)
        feats_input = self.net(input_scaled)
        feats_target = self.net(target_scaled)
        diffs = []
        lin_layers = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        for idx, (feat_in, feat_tgt) in enumerate(zip(feats_input, feats_target)):
            feat_in = _normalize(feat_in)
            feat_tgt = _normalize(feat_tgt)
            diff = (feat_in - feat_tgt) ** 2
            diffs.append(_spatial_average(lin_layers[idx].model(diff), keepdim=True))

        value = diffs[0]
        for diff in diffs[1:]:
            value = value + diff

        if reduction == "none":
            return value
        if reduction == "sum":
            return torch.sum(value)
        if reduction == "mean":
            return torch.mean(value)
        raise ValueError(f"Unsupported reduction '{reduction}'")
