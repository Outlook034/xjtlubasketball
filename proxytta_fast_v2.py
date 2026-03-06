import copy
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from completion_tta import (
    sparse_depth_consistency_loss_func,
    smoothness_loss_func,
    OutlierRemoval,
    Transforms,
    create_sparse_depth,
)


def _normalize_img(img: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    if img.max() > 1.0:
        img = img / 255.0
    return (img - mean) / std


@torch.no_grad()
def ema_update(teacher: nn.Module, student: nn.Module, momentum: float) -> None:
    for t_param, s_param in zip(teacher.parameters(), student.parameters()):
        t_param.data.mul_(momentum).add_(s_param.data, alpha=1.0 - momentum)


class SQLDepthFeatureModel(nn.Module):
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
class ProxyTTAFastV2Config:
    steps: int = 1
    lr: float = 1e-4
    momentum_teacher: float = 0.999
    w_sparse: float = 1.0
    w_smooth: float = 1.0
    w_cos: float = 1.0
    grad_clip: float = 5.0
    norm_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    norm_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    # Augmentation (mirrors CompletionTTAModule)
    aug_prob: float = 0.5
    aug_flip_type: list = None
    aug_rotate: float = 0.0
    aug_brightness: list = None
    aug_contrast: list = None
    aug_gamma: list = None
    aug_saturation: list = None
    aug_noise_type: str = 'none'
    aug_noise_spread: float = -1

    def __post_init__(self):
        if self.aug_flip_type is None:
            self.aug_flip_type = ['none']
        if self.aug_brightness is None:
            self.aug_brightness = [-1, -1]
        if self.aug_contrast is None:
            self.aug_contrast = [-1, -1]
        if self.aug_gamma is None:
            self.aug_gamma = [-1, -1]
        if self.aug_saturation is None:
            self.aug_saturation = [-1, -1]


class ProxyTTAFastV2Module(nn.Module):
    """
    ProxyTTA-fast v2:
    - Uses the same adaptation signals as Completion-TTA (sparse depth consistency + smoothness)
    - Updates ONLY the adaptation layer (fast)
    - Optionally adds proxy cosine regularization (EMA teacher feature alignment)
    - Mirrors CompletionTTAModule's augmentation pipeline
    """

    def __init__(self, depth_model: SQLDepthFeatureModel, config: ProxyTTAFastV2Config):
        super().__init__()
        self.student = depth_model
        self.config = config

        # Teacher (EMA copy)
        self.teacher = copy.deepcopy(depth_model)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        # Freeze all student params
        for p in self.student.parameters():
            p.requires_grad = False

        # Adaptation layer (lazy init)
        self.adapt_layer: Optional[AdaptationLayer] = None

        # Augmentation (mirrors CompletionTTAModule)
        self.outlier_removal = OutlierRemoval(7, 1.5)
        self.train_geom = Transforms(
            random_crop_to_shape=[-1, -1],
            random_flip_type=config.aug_flip_type,
            random_rotate_max=config.aug_rotate,
            random_crop_and_pad=[-1, -1],
            random_resize_and_pad=[-1, -1],
            random_resize_and_crop=[-1, -1]
        )
        self.interpolation_modes = self.train_geom.map_interpolation_mode_names_to_enums(
            ['bilinear', 'nearest', 'nearest']
        )
        self.train_photo = Transforms(
            normalized_image_range=[0, 255],
            random_brightness=config.aug_brightness,
            random_contrast=config.aug_contrast,
            random_gamma=config.aug_gamma,
            random_hue=[-1, -1],
            random_saturation=config.aug_saturation,
            random_noise_type=config.aug_noise_type,
            random_noise_spread=config.aug_noise_spread
        )

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

    def _proxy_cos_loss(self, student_feat: torch.Tensor, teacher_feat: torch.Tensor) -> torch.Tensor:
        s = F.normalize(student_feat, p=2, dim=1)
        t = F.normalize(teacher_feat, p=2, dim=1)
        cos = (s * t).sum(dim=1, keepdim=True)
        return (2.0 - 2.0 * cos).mean()

    @torch.enable_grad()
    def adapt(self, image: torch.Tensor, sparse_depth: torch.Tensor) -> torch.Tensor:
        """Run test-time adaptation with sparse depth consistency + smoothness + optional proxy cosine."""
        self.student.eval()
        self.teacher.eval()

        # Normalize input once
        image_norm = _normalize_img(image, self.norm_mean, self.norm_std)

        # Teacher feature (reference)
        with torch.no_grad():
            _, teacher_feat = self.teacher(image_norm)

        # Initialize adaptation layer
        with torch.no_grad():
            _, student_feat0 = self.student(image_norm)
        self._ensure_adapt_layer(student_feat0)
        assert self.adapt_layer is not None and self.optimizer is not None

        # Sparse depth preprocessing (mirrors CompletionTTAModule)
        validity = (sparse_depth > 0).float()
        filtered_sparse, filtered_valid = self.outlier_removal.remove_outliers(
            sparse_depth=sparse_depth,
            validity_map=validity
        )

        # Augmentation (mirrors CompletionTTAModule)
        geom_outputs = self.train_geom.transform(
            images_arr=[image_norm, filtered_sparse, filtered_valid],
            intrinsics_arr=[],
            interpolation_modes=self.interpolation_modes,
            random_transform_probability=self.config.aug_prob
        )
        image_geom, sparse_geom, valid_geom = geom_outputs

        # Resize sparse/valid to match image_geom
        sparse_geom = torch.nn.functional.interpolate(
            sparse_geom, size=image_geom.shape[2:], mode='nearest'
        )
        valid_geom = torch.nn.functional.interpolate(
            valid_geom, size=image_geom.shape[2:], mode='nearest'
        )

        [image_photo] = self.train_photo.transform(
            images_arr=[image_geom],
            random_transform_probability=self.config.aug_prob
        )

        # Optimize only adaptation layer
        for _ in range(self.config.steps):
            # Student forward
            _, feat = self.student(image_photo)

            # Residual adaptation (only this layer is trainable)
            delta = self.adapt_layer(feat)
            feat_adapted = feat + delta

            # Decode using adapted feature
            disp = self.student.decode(feat_adapted)

            # Resize sparse/valid to match disp if needed
            if sparse_geom.shape[2:] != disp.shape[2:]:
                sparse_for_loss = torch.nn.functional.interpolate(
                    sparse_geom, size=disp.shape[2:], mode='nearest'
                )
                valid_for_loss = torch.nn.functional.interpolate(
                    valid_geom, size=disp.shape[2:], mode='nearest'
                )
                image_for_loss = torch.nn.functional.interpolate(
                    image_photo, size=disp.shape[2:], mode='bilinear', align_corners=False
                )
            else:
                sparse_for_loss = sparse_geom
                valid_for_loss = valid_geom
                image_for_loss = image_photo

            # Losses (mirrors CompletionTTAModule + optional proxy cosine)
            loss_sparse = sparse_depth_consistency_loss_func(
                disp, sparse_for_loss, valid_for_loss
            )
            loss_smooth = smoothness_loss_func(
                predict=disp,
                image=image_for_loss / 255.0 if image_for_loss.max() > 1.0 else image_for_loss
            )
            loss_cos = self._proxy_cos_loss(feat_adapted, teacher_feat.detach())

            total_loss = (
                self.config.w_sparse * loss_sparse +
                self.config.w_smooth * loss_smooth +
                self.config.w_cos * loss_cos
            )

            self.optimizer.zero_grad()
            total_loss.backward()
            if self.config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.adapt_layer.parameters(), self.config.grad_clip)
            self.optimizer.step()

            # EMA update teacher
            with torch.no_grad():
                ema_update(self.teacher, self.student, self.config.momentum_teacher)

        # Final inference using adapted feature
        self.student.eval()
        with torch.no_grad():
            _, feat_final = self.student(image_norm)
            feat_final = feat_final + self.adapt_layer(feat_final)
            final_disp = self.student.decode(feat_final)
        return final_disp