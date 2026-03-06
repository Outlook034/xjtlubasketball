from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _extract_tensor(feature_map):
    """从多种结构中提取最后的 Tensor。"""
    if isinstance(feature_map, torch.Tensor):
        return feature_map

    if isinstance(feature_map, dict):
        for value in feature_map.values():
            if torch.is_tensor(value):
                return value
        raise ValueError("特征字典中未找到 Tensor。")

    if isinstance(feature_map, (list, tuple)):
        for item in reversed(feature_map):
            if torch.is_tensor(item):
                return item
        raise ValueError("特征列表中未找到 Tensor。")

    raise TypeError(f"无法从类型 {type(feature_map).__name__} 中提取 Tensor。")


class FeatureProjector(nn.Module):
    """将任意形状的特征映射压缩成 (B, D) 向量。"""

    def __init__(self, pool_type: str = "avg"):
        super().__init__()
        if pool_type not in {"avg", "max"}:
            raise ValueError(f"不支持的池化方式: {pool_type}")
        self.pool_type = pool_type

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        feat = _extract_tensor(feature_map)
        if feat.dim() > 2:
            if self.pool_type == "avg":
                feat = F.adaptive_avg_pool2d(feat, output_size=1)
            else:
                feat = F.adaptive_max_pool2d(feat, output_size=1)
        return feat.flatten(1)


def diagonal_gaussian_kl_loss(m1: torch.Tensor, v1: torch.Tensor,
                              m2: torch.Tensor, v2: torch.Tensor,
                              eps: float = 0.0,
                              reduction: str = "none") -> torch.Tensor:
    """逐维 KL 散度，和 regression-tta 实现保持一致。"""
    loss = (v2.add(eps).log() - v1.add(eps).log() +
            (v1 + (m2 - m1).square()) / (v2 + eps) - 1) * 0.5
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    if reduction == "none":
        return loss
    raise ValueError(f"非法 reduction: {reduction}")


def load_pca_stats(stat_path: str, topk: Optional[int]):
    """加载 PCA 统计量并选取贡献度最高的前 topk 个主成分。"""
    stats = torch.load(stat_path, map_location="cpu")
    mean = stats["mean"].float()
    basis = stats["basis"].float()
    eigvals = stats["eigvals"].float()

    if topk is not None and topk > 0:
        topk = min(topk, eigvals.shape[0])
        _, indices = torch.topk(eigvals, k=topk)
        basis = basis[:, indices]
        eigvals = eigvals[indices]

    return mean, basis, eigvals


def build_dim_weight(var: torch.Tensor, bias: float, exponent: float) -> torch.Tensor:
    """基于特征方差生成维度权重，避免数值为 0。"""
    normed = var / var.mean().clamp_min(1e-6)
    weight = normed.abs() + bias
    if exponent != 1.0:
        weight = weight.pow(exponent)
    return weight

