import copy
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _normalize_img(img: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    if img.max() > 1.0:
        img = img / 255.0
    return (img - mean) / std


@torch.no_grad()
def ema_update(teacher: nn.Module, student: nn.Module, momentum: float) -> None:
    """EMA update teacher parameters from student."""
    for t_param, s_param in zip(teacher.parameters(), student.parameters()):
        t_param.data.mul_(momentum).add_(s_param.data, alpha=1.0 - momentum)


class SQLDepthFeatureModel(nn.Module):
    """Wrap encoder+decoder so we can extract a stable feature map for proxy loss.

    We use the *encoder output* (the tensor that is fed into Depth_Decoder_QueryTr)
    as the feature. In this codebase, that is `encoder(images_norm)` where `encoder`
    is e.g. `Resnet50EncoderDecoder`.

    Note: the decoder returns a dict with ("disp", 0).
    """

    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def decode(self, feat: torch.Tensor) -> torch.Tensor:
        out = self.decoder(feat)
        if isinstance(out, dict):
            return out[("disp", 0)]
        return out

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.encoder(x)
        disp = self.decode(feat)
        return disp, feat


class AdaptationLayer(nn.Module):
    """A lightweight adaptation layer applied on top of encoder feature map."""

    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class ProxyTTAFastConfig:
    steps: int = 1
    lr: float = 1e-4
    momentum_teacher: float = 0.999
    w_cos: float = 1.0
    w_smooth: float = 1.0
    grad_clip: float = 5.0
    norm_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    norm_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)


class ProxyTTAFastModule(nn.Module):
    """ProxyTTA-fast for monocular depth estimation using an EMA teacher.

    - Student: SQLDepth model (encoder+decoder)
    - Teacher: EMA copy of student (frozen)
    - Adaptation: only update `adapt_layer` (fast)

    Proxy loss: cosine distance between adapted student features and teacher features.
    Smoothness loss: edge-aware smoothness on predicted disp.

    This is designed to be a *drop-in replacement* for your existing TTA module,
    while keeping the core depth estimation task unchanged.
    """

    def __init__(self, depth_model: SQLDepthFeatureModel, config: ProxyTTAFastConfig):
        super().__init__()
        self.student = depth_model
        self.config = config

        # Build teacher as EMA copy
        self.teacher = copy.deepcopy(depth_model)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        # Freeze all student params
        for p in self.student.parameters():
            p.requires_grad = False

        # Infer feature channel dim for adaptation layer
        # Use a lazy conv-like init: create layer after first forward if needed.
        self.adapt_layer: Optional[AdaptationLayer] = None

        mean = torch.tensor(config.norm_mean).view(1, 3, 1, 1)
        std = torch.tensor(config.norm_std).view(1, 3, 1, 1)
        self.register_buffer("norm_mean", mean)
        self.register_buffer("norm_std", std)

        self.optimizer: Optional[torch.optim.Optimizer] = None

    def _ensure_adapt_layer(self, feat: torch.Tensor) -> None:
        if self.adapt_layer is not None:
            return
        if feat.dim() != 4:
            raise ValueError(f"Expected 4D feature map (B,C,H,W), got shape {tuple(feat.shape)}")
        channels = feat.shape[1]
        self.adapt_layer = AdaptationLayer(channels).to(feat.device)
        for p in self.adapt_layer.parameters():
            p.requires_grad = True
        self.optimizer = torch.optim.Adam(self.adapt_layer.parameters(), lr=self.config.lr)

    def _smoothness_loss(self, disp: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        # Edge-aware smoothness (Monodepth2-style)
        # Ensure image is at the same spatial resolution as disp
        if img.shape[-2:] != disp.shape[-2:]:
            img = F.interpolate(img, size=disp.shape[-2:], mode='bilinear', align_corners=False)

        def gradient_x(t):
            return t[:, :, :, 1:] - t[:, :, :, :-1]

        def gradient_y(t):
            return t[:, :, 1:, :] - t[:, :, :-1, :]

        disp_grad_x = gradient_x(disp)
        disp_grad_y = gradient_y(disp)

        img_gray = img.mean(1, keepdim=True)
        img_grad_x = gradient_x(img_gray)
        img_grad_y = gradient_y(img_gray)

        weight_x = torch.exp(-torch.abs(img_grad_x))
        weight_y = torch.exp(-torch.abs(img_grad_y))

        smooth_x = (torch.abs(disp_grad_x) * weight_x).mean()
        smooth_y = (torch.abs(disp_grad_y) * weight_y).mean()
        return smooth_x + smooth_y

    def _proxy_cos_loss(self, student_feat: torch.Tensor, teacher_feat: torch.Tensor) -> torch.Tensor:
        # Compare per-pixel features in cosine space
        # Normalize along channel dim
        s = F.normalize(student_feat, p=2, dim=1)
        t = F.normalize(teacher_feat, p=2, dim=1)
        # cosine distance: 2 - 2*cos
        cos = (s * t).sum(dim=1, keepdim=True)
        return (2.0 - 2.0 * cos).mean()

    @torch.enable_grad()
    def adapt(self, image: torch.Tensor) -> torch.Tensor:
        """Run test-time adaptation on a batch and return final prediction."""
        self.student.eval()  # keep BN frozen etc.
        self.teacher.eval()

        # Normalize input once
        image_norm = _normalize_img(image, self.norm_mean, self.norm_std)

        # Get teacher feature (reference)
        with torch.no_grad():
            teacher_disp, teacher_feat = self.teacher(image_norm)

        # Student forward to initialize adapt layer
        with torch.no_grad():
            _, student_feat0 = self.student(image_norm)
        self._ensure_adapt_layer(student_feat0)
        assert self.adapt_layer is not None and self.optimizer is not None

        # Optimize only adaptation layer
        for _ in range(self.config.steps):
            # Student feature
            _, feat = self.student(image_norm)

            # Residual adaptation (A): feat_adapted = feat + adapt_layer(feat)
            delta = self.adapt_layer(feat)
            feat_adapted = feat + delta

            # Decode using adapted feature so adaptation affects prediction
            disp = self.student.decode(feat_adapted)

            loss_cos = self._proxy_cos_loss(feat_adapted, teacher_feat.detach())
            loss_smooth = self._smoothness_loss(disp, image_norm)
            total_loss = self.config.w_cos * loss_cos + self.config.w_smooth * loss_smooth

            self.optimizer.zero_grad()
            total_loss.backward()
            if self.config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.adapt_layer.parameters(), self.config.grad_clip)
            self.optimizer.step()

            # Update teacher (EMA)
            with torch.no_grad():
                ema_update(self.teacher, self.student, self.config.momentum_teacher)

        # Final inference using adapted feature
        self.student.eval()
        with torch.no_grad():
            _, feat_final = self.student(image_norm)
            feat_final = feat_final + self.adapt_layer(feat_final)
            final_disp = self.student.decode(feat_final)
        return final_disp
