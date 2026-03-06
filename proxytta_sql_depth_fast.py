import copy
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from loss_utils import sparse_depth_consistency_loss_func, smoothness_loss_func
from transforms import Transforms
from net_utils import OutlierRemoval


def _normalize_img(img: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    if img.max() > 1.0:
        img = img / 255.0
    return (img - mean) / std


class SQLDepthModel(nn.Module):
    """Encoder+Decoder wrapper that exposes encoder features."""

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


class SparseDepthEncoder(nn.Module):
    """Lightweight sparse depth encoder to produce proxy modality features."""

    def __init__(self, out_channels: int):
        super().__init__()
        mid = max(16, out_channels // 4)
        self.net = nn.Sequential(
            nn.Conv2d(1, mid, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, out_channels, kernel_size=3, stride=2, padding=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EmbeddingModule(nn.Module):
    """Maps sparse-depth features to proxy embeddings in RGB feature space."""

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
class ProxyTTASQLDepthFastConfig:
    # Adapt steps
    steps: int = 1
    lr: float = 4e-5
    grad_clip: float = 5.0

    # Loss weights
    w_sparse: float = 1.0
    w_smooth: float = 1.0
    w_cos: float = 1.0

    # Only update embedding module by default (fast)
    update_sparse_encoder: bool = False

    # Normalization
    norm_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    norm_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    # Augmentation (mirror completion_tta.py)
    aug_prob: float = 0.0
    aug_flip_type: Tuple[str, ...] = ("none",)
    aug_rotate: float = 0.0
    aug_brightness: Tuple[float, float] = (-1.0, -1.0)
    aug_contrast: Tuple[float, float] = (-1.0, -1.0)
    aug_gamma: Tuple[float, float] = (-1.0, -1.0)
    aug_saturation: Tuple[float, float] = (-1.0, -1.0)
    aug_noise_type: str = "none"
    aug_noise_spread: float = -1.0

    # For geometric crop transform
    height: int = 192
    width: int = 640


class ProxyTTASQLDepthFastModule(nn.Module):
    """ProxyTTA-fast adapted to SQLDepth, following the paper structure.

    - embedding: RGB branch high-level feature (encoder output)
    - reference: proxy embedding generated from sparse depth branch feature via EmbeddingModule

    Fast: only updates EmbeddingModule (and optionally sparse-depth encoder).
    """

    def __init__(self, depth_model: SQLDepthModel, cfg: ProxyTTASQLDepthFastConfig):
        super().__init__()
        self.depth_model = depth_model
        self.cfg = cfg

        # Freeze base model
        for p in self.depth_model.parameters():
            p.requires_grad = False

        mean = torch.tensor(cfg.norm_mean).view(1, 3, 1, 1)
        std = torch.tensor(cfg.norm_std).view(1, 3, 1, 1)
        self.register_buffer("norm_mean", mean)
        self.register_buffer("norm_std", std)

        self.outlier_removal = OutlierRemoval(7, 1.5)
        self.train_geom = Transforms(
            random_crop_to_shape=[cfg.height, cfg.width],
            random_flip_type=list(cfg.aug_flip_type),
            random_rotate_max=cfg.aug_rotate,
            random_crop_and_pad=[-1, -1],
            random_resize_and_pad=[-1, -1],
            random_resize_and_crop=[-1, -1],
        )
        self.interpolation_modes = self.train_geom.map_interpolation_mode_names_to_enums(
            ["bilinear", "nearest", "nearest"]
        )
        self.train_photo = Transforms(
            normalized_image_range=[0, 255],
            random_brightness=list(cfg.aug_brightness),
            random_contrast=list(cfg.aug_contrast),
            random_gamma=list(cfg.aug_gamma),
            random_hue=[-1, -1],
            random_saturation=list(cfg.aug_saturation),
            random_noise_type=cfg.aug_noise_type,
            random_noise_spread=cfg.aug_noise_spread,
        )

        self.sparse_encoder: Optional[SparseDepthEncoder] = None
        self.embedding_module: Optional[EmbeddingModule] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None

    def _ensure_modules(self, rgb_feat: torch.Tensor) -> None:
        if self.sparse_encoder is not None and self.embedding_module is not None:
            return
        c = int(rgb_feat.shape[1])
        self.sparse_encoder = SparseDepthEncoder(out_channels=c).to(rgb_feat.device)
        self.embedding_module = EmbeddingModule(channels=c).to(rgb_feat.device)

        params = list(self.embedding_module.parameters())
        if self.cfg.update_sparse_encoder:
            params += list(self.sparse_encoder.parameters())
        self.optimizer = torch.optim.Adam(params, lr=self.cfg.lr)

    def _proxy_cos_loss(self, embedding: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        e = F.normalize(embedding, p=2, dim=1)
        r = F.normalize(reference, p=2, dim=1)
        cos = (e * r).sum(dim=1, keepdim=True)
        return (2.0 - 2.0 * cos).mean()

    @torch.enable_grad()
    def adapt(self, image: torch.Tensor, sparse_depth: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """Returns (final_disp, debug_info)."""
        self.depth_model.eval()

        validity = (sparse_depth > 0).float()
        filtered_sparse, filtered_valid = self.outlier_removal.remove_outliers(
            sparse_depth=sparse_depth,
            validity_map=validity,
        )

        geom_outputs = self.train_geom.transform(
            images_arr=[image, filtered_sparse, filtered_valid],
            intrinsics_arr=[],
            interpolation_modes=self.interpolation_modes,
            random_transform_probability=self.cfg.aug_prob,
        )
        image_geom, sparse_geom, valid_geom = geom_outputs

        sparse_geom = F.interpolate(sparse_geom, size=image_geom.shape[2:], mode="nearest")
        valid_geom = F.interpolate(valid_geom, size=image_geom.shape[2:], mode="nearest")

        [image_photo] = self.train_photo.transform(
            images_arr=[image_geom],
            random_transform_probability=self.cfg.aug_prob,
        )

        debug = {
            "loss_sparse": 0.0,
            "loss_smooth": 0.0,
            "loss_cos": 0.0,
            "loss_total": 0.0,
        }

        for _ in range(self.cfg.steps):
            # Normalize image for RGB encoder
            img_norm = _normalize_img(image_photo, self.norm_mean, self.norm_std)

            # RGB embedding
            disp, rgb_feat = self.depth_model(img_norm)
            self._ensure_modules(rgb_feat)
            assert self.sparse_encoder is not None and self.embedding_module is not None and self.optimizer is not None

            # Proxy reference from sparse depth
            sd_feat = self.sparse_encoder(sparse_geom)
            if sd_feat.shape[2:] != rgb_feat.shape[2:]:
                sd_feat = F.interpolate(sd_feat, size=rgb_feat.shape[2:], mode="bilinear", align_corners=False)
            ref = self.embedding_module(sd_feat)

            # Losses
            # Resize sparse/valid to match disp
            sparse_for_loss = sparse_geom
            valid_for_loss = valid_geom
            if sparse_for_loss.shape[2:] != disp.shape[2:]:
                sparse_for_loss = F.interpolate(sparse_for_loss, size=disp.shape[2:], mode="nearest")
                valid_for_loss = F.interpolate(valid_for_loss, size=disp.shape[2:], mode="nearest")

            img_for_smooth = image_photo
            if img_for_smooth.shape[2:] != disp.shape[2:]:
                img_for_smooth = F.interpolate(img_for_smooth, size=disp.shape[2:], mode="bilinear", align_corners=False)

            loss_sparse = sparse_depth_consistency_loss_func(disp, sparse_for_loss, valid_for_loss)
            loss_smooth = smoothness_loss_func(
                predict=disp,
                image=img_for_smooth / 255.0 if img_for_smooth.max() > 1.0 else img_for_smooth,
            )
            loss_cos = self._proxy_cos_loss(rgb_feat, ref)

            total = self.cfg.w_sparse * loss_sparse + self.cfg.w_smooth * loss_smooth + self.cfg.w_cos * loss_cos

            self.optimizer.zero_grad()
            total.backward()
            if self.cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]["params"], self.cfg.grad_clip)
            self.optimizer.step()

            debug["loss_sparse"] = float(loss_sparse.detach().item())
            debug["loss_smooth"] = float(loss_smooth.detach().item())
            debug["loss_cos"] = float(loss_cos.detach().item())
            debug["loss_total"] = float(total.detach().item())

        # Final prediction on original image
        with torch.no_grad():
            img_norm = _normalize_img(image, self.norm_mean, self.norm_std)
            final_disp, _ = self.depth_model(img_norm)

        return final_disp, debug
