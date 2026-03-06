import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import time
import warnings

# 过滤警告
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")

from layers import disp_to_depth, BackprojectDepth, Project3D, transformation_from_parameters
from utils import readlines, normalize_image
from SQLdepth import SQLdepth, MonodepthOptions
import networks
from ssa_utils import (
    FeatureProjector,
    load_pca_stats,
    diagonal_gaussian_kl_loss,
    build_dim_weight,
)

# 1. SQLdepth 模型包装器，用于 TTA
class SQLDepthModel(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, x):
        out = self.decoder(self.encoder(x))
        return out[("disp", 0)]  # 只返回主Tensor

# 2. VECTTA: 视角等变一致TTA
class VECTTA(nn.Module):
    """几何等变一致测试时适应 (Viewpoint Equivariance Consistency Test-Time Adaptation)
    
    通过注入虚拟微小相机位姿变换，构建深度预测的几何闭环一致性约束。
    """
    def __init__(self, depth_model, update_mode='bn_decoder', optimizer_lr=1e-4, steps=5,
                 early_stop=True, early_stop_patience=3, grad_clip=1.0):
        super().__init__()
        self.depth_model = depth_model
        self.update_mode = update_mode
        self.optimizer_lr = optimizer_lr
        self.steps = steps
        self.early_stop = early_stop
        self.early_stop_patience = early_stop_patience
        self.grad_clip = grad_clip

        self.configure_params_to_update()
        self.optimizer = torch.optim.AdamW(self.params_to_update, lr=optimizer_lr)

    def configure_params_to_update(self):
        self.params_to_update = []
        for p in self.depth_model.parameters():
            p.requires_grad = False
            
        if self.update_mode == 'all':
            # 更新整个模型的所有参数
            for p in self.depth_model.parameters():
                p.requires_grad = True
                self.params_to_update.append(p)

        if self.update_mode == 'bn_only':
            for m in self.depth_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.requires_grad_(True)
                    m.track_running_stats = False
                    m.running_mean = None
                    m.running_var = None
                    for p in m.parameters():
                        p.requires_grad = True
                        self.params_to_update.append(p)

        if self.update_mode == 'bn_decoder':
            if hasattr(self.depth_model, 'decoder'):
                for p in self.depth_model.decoder.parameters():
                    p.requires_grad = True
                    self.params_to_update.append(p)
            # 同时解冻所有BN层
            for m in self.depth_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.requires_grad_(True)
                    m.track_running_stats = False
                    m.running_mean = None
                    m.running_var = None
                    for p in m.parameters():
                        p.requires_grad = True
                        self.params_to_update.append(p)

        if self.update_mode == 'last_layers':
            # 仅示例：匹配名称里包含"disp"的Conv层
            for name, m in self.depth_model.named_modules():
                if isinstance(m, nn.Conv2d) and 'disp' in name:
                    for p in m.parameters():
                        p.requires_grad = True
                        self.params_to_update.append(p)
            # 同时放开BN
            for m in self.depth_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.requires_grad_(True)
                    m.track_running_stats = False
                    m.running_mean = None
                    m.running_var = None
                    for p in m.parameters():
                        p.requires_grad = True
                        self.params_to_update.append(p)

    def sample_virtual_poses(self, batch_size, device=None, num_poses=4, xyz_range=0.1, rpy_range=0.05):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        patterns = [
            [xyz_range, 0, 0, 0, rpy_range, 0],
            [-xyz_range, 0, 0, 0, -rpy_range, 0],
            [0, 0, xyz_range, rpy_range, 0, 0],
            [0, xyz_range, 0, 0, 0, rpy_range],
        ]
        poses_list = []
        for tx, ty, tz, rx, ry, rz in patterns:
            translation = torch.zeros(batch_size, 1, 1, 3, device=device)
            translation[:, 0, 0, 0] = tx
            translation[:, 0, 0, 1] = ty
            translation[:, 0, 0, 2] = tz
            axisangle = torch.zeros(batch_size, 1, 1, 3, device=device)
            axisangle[:, 0, 0, 0] = rx
            axisangle[:, 0, 0, 1] = ry
            axisangle[:, 0, 0, 2] = rz
            poses_list.append((axisangle, translation))
        return poses_list

    def back_project_depth(self, depth, K, pose):
        batch_size, _, height, width = depth.shape
        device = depth.device
        K_3x3 = K[:, :3, :3] if K.shape[-1] == 4 else K
        backproject = BackprojectDepth(batch_size, height, width).to(device)
        project = Project3D(batch_size, height, width).to(device)
        inv_K = torch.inverse(K_3x3)
        K_proj = torch.zeros(batch_size, 3, 4, device=device)
        K_proj[:, :3, :3] = K_3x3
        axisangle, translation = pose
        T = transformation_from_parameters(axisangle[:, 0], translation[:, 0], invert=True)
        cam_points = backproject(depth, inv_K)
        pix_coords = project(cam_points, K_proj, T)
        z = cam_points[:, 2, :].reshape(batch_size, 1, height, width)
        transformed_depth = F.grid_sample(z, pix_coords, mode='bilinear', padding_mode='zeros', align_corners=True)
        valid_mask = ((pix_coords[:, :, :, 0] >= -1) & (pix_coords[:, :, :, 0] <= 1) &
                      (pix_coords[:, :, :, 1] >= -1) & (pix_coords[:, :, :, 1] <= 1)).unsqueeze(1).float()
        return transformed_depth * valid_mask

    def create_visibility_mask(self, depth, back_depth, border_margin=2):
        valid = torch.isfinite(back_depth) & (back_depth > 1e-6) & torch.isfinite(depth) & (depth > 1e-6)
        mask = valid.float()
        if border_margin > 0:
            mask[:, :, :border_margin, :] = 0
            mask[:, :, -border_margin:, :] = 0
            mask[:, :, :, :border_margin] = 0
            mask[:, :, :, -border_margin:] = 0
        return mask

    def compute_scale_invariant_depth_error(self, depth, back_depth, mask):
        eps = 1e-6
        depth_safe = torch.clamp(depth, min=eps)
        back_depth_safe = torch.clamp(back_depth, min=eps)
        if mask is None:
            mask = torch.ones_like(depth_safe)
        valid = mask > 0.5
        if valid.sum() == 0:
            return torch.zeros((), device=depth.device)
        log_diff = (torch.log(depth_safe) - torch.log(back_depth_safe))[valid]
        n = log_diff.numel()
        loss = (log_diff.pow(2).sum() / n) - (log_diff.sum() ** 2) / (n ** 2)
        return loss

    def compute_gradient_consistency_loss(self, depth, back_depth, mask):
        if mask is None:
            mask = torch.ones_like(depth)
        def gradient_x(img):
            return img[:, :, :, 1:] - img[:, :, :, :-1]
        def gradient_y(img):
            return img[:, :, 1:, :] - img[:, :, :-1, :]
        gx1, gy1 = gradient_x(depth), gradient_y(depth)
        gx2, gy2 = gradient_x(back_depth), gradient_y(back_depth)
        mask_x = mask[:, :, :, 1:] * mask[:, :, :, :-1]
        mask_y = mask[:, :, 1:, :] * mask[:, :, :-1, :]
        diff_x = torch.abs(gx1 - gx2) * mask_x
        diff_y = torch.abs(gy1 - gy2) * mask_y
        denom = (mask_x.sum() + mask_y.sum()).clamp(min=1.0)
        return (diff_x.sum() + diff_y.sum()) / denom

    def forward(self, image, K, clean_depth=None):
        device = image.device
        batch_size = image.shape[0]
        self.depth_model.train()
        best_loss = float('inf')
        patience_counter = 0
        
        for step_idx in range(self.steps):
            # 需要梯度：深度必须可导，才能把几何一致性损失反传到模型参数
            depth = self.depth_model(image)
            if isinstance(depth, dict):
                depth = depth[("disp", 0)]
            poses = self.sample_virtual_poses(batch_size, device)
            total_loss = depth.new_tensor(0.0)
            for pose in poses:
                # 基于原视角深度，生成新视角"回投影"深度，避免重复图像渲染
                back_depth = self.back_project_depth(depth, K, pose)
                mask = self.create_visibility_mask(depth, back_depth)
                si_loss = self.compute_scale_invariant_depth_error(depth, back_depth, mask)
                grad_loss = self.compute_gradient_consistency_loss(depth, back_depth, mask)
                total_loss = total_loss + si_loss + 0.5 * grad_loss
            total_loss = total_loss / max(len(poses), 1)
            
            clean_loss = 0.0
            if clean_depth is not None:
                if clean_depth.dim() == 3:
                    clean_depth = clean_depth.unsqueeze(1)
                
                # 确保clean_depth和depth的形状匹配
                if clean_depth.shape != depth.shape:
                    # 调整clean_depth的形状以匹配depth
                    if clean_depth.shape[0] != depth.shape[0]:
                        min_batch_size = min(clean_depth.shape[0], depth.shape[0])
                        clean_depth = clean_depth[:min_batch_size]
                        depth = depth[:min_batch_size]
                    
                    # 调整空间维度
                    if clean_depth.shape[2:] != depth.shape[2:]:
                        clean_depth = F.interpolate(clean_depth, size=depth.shape[2:], mode='bilinear', align_corners=False)
                
                # 只有当clean_depth不为空时才计算损失
                if clean_depth.numel() > 0 and depth.numel() > 0:
                    clean_loss = F.l1_loss(depth, clean_depth)
                    total_loss = total_loss + 0.5 * clean_loss
            
            self.optimizer.zero_grad()
            total_loss.backward()
            if self.grad_clip is not None and len(self.params_to_update) > 0:
                torch.nn.utils.clip_grad_norm_(self.params_to_update, self.grad_clip)
            self.optimizer.step()
            
            if self.early_stop:
                loss_val = float(total_loss.detach().item())
                if loss_val < best_loss:
                    best_loss = loss_val
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= self.early_stop_patience:
                    break
        
        self.depth_model.eval()
        with torch.no_grad():
            final_depth = self.depth_model(image)
            if isinstance(final_depth, dict):
                final_depth = final_depth[("disp", 0)]
        return final_depth

# 2.5. VecTTAFast: VecTTA的Proxy-fast变体，冻结所有模型参数，使用代理参数进行适应
class VecTTAFast(nn.Module):
    """VecTTA Fast变体：冻结所有模型参数，使用代理参数进行适应
    
    类似于Proxy-fast方法，通过代理参数来适应，保持原始模型参数完全冻结。
    这样可以避免对原始模型的直接修改，提高稳定性。
    """
    def __init__(self, depth_model, update_mode='bn_decoder', optimizer_lr=1e-4, steps=5,
                 early_stop=True, early_stop_patience=3, grad_clip=1.0):
        super().__init__()
        self.depth_model = depth_model
        self.update_mode = update_mode
        self.optimizer_lr = optimizer_lr
        self.steps = steps
        self.early_stop = early_stop
        self.early_stop_patience = early_stop_patience
        self.grad_clip = grad_clip

        # 冻结所有原始模型参数
        for p in self.depth_model.parameters():
            p.requires_grad = False
        
        # 创建代理参数
        self.proxy_params = {}
        self.param_mapping = {}  # 存储原始参数到代理参数的映射
        self.create_proxy_parameters()
        
        # 只优化代理参数
        self.params_to_update = list(self.proxy_params.values())
        self.optimizer = torch.optim.AdamW(self.params_to_update, lr=optimizer_lr)
    
    def create_proxy_parameters(self):
        """为需要更新的参数创建代理参数"""
        if self.update_mode == 'all':
            # 为所有参数创建代理
            for name, param in self.depth_model.named_parameters():
                proxy_param = nn.Parameter(param.data.clone())
                self.proxy_params[name] = proxy_param
                self.param_mapping[name] = param
        
        elif self.update_mode == 'bn_only':
            # 只为BN层创建代理参数
            for name, module in self.depth_model.named_modules():
                if isinstance(module, nn.BatchNorm2d):
                    # 为BN的weight和bias创建代理
                    if hasattr(module, 'weight') and module.weight is not None:
                        proxy_name = f"{name}.weight"
                        proxy_param = nn.Parameter(module.weight.data.clone())
                        self.proxy_params[proxy_name] = proxy_param
                        self.param_mapping[proxy_name] = module.weight
                    if hasattr(module, 'bias') and module.bias is not None:
                        proxy_name = f"{name}.bias"
                        proxy_param = nn.Parameter(module.bias.data.clone())
                        self.proxy_params[proxy_name] = proxy_param
                        self.param_mapping[proxy_name] = module.bias
                    # 配置BN层
                    module.track_running_stats = False
                    module.running_mean = None
                    module.running_var = None
        
        elif self.update_mode == 'bn_decoder':
            # 为decoder和BN层创建代理参数
            if hasattr(self.depth_model, 'decoder'):
                for name, param in self.depth_model.decoder.named_parameters():
                    full_name = f"decoder.{name}"
                    proxy_param = nn.Parameter(param.data.clone())
                    self.proxy_params[full_name] = proxy_param
                    self.param_mapping[full_name] = param
            
            # 为所有BN层创建代理参数
            for name, module in self.depth_model.named_modules():
                if isinstance(module, nn.BatchNorm2d):
                    if hasattr(module, 'weight') and module.weight is not None:
                        proxy_name = f"{name}.weight"
                        proxy_param = nn.Parameter(module.weight.data.clone())
                        self.proxy_params[proxy_name] = proxy_param
                        self.param_mapping[proxy_name] = module.weight
                    if hasattr(module, 'bias') and module.bias is not None:
                        proxy_name = f"{name}.bias"
                        proxy_param = nn.Parameter(module.bias.data.clone())
                        self.proxy_params[proxy_name] = proxy_param
                        self.param_mapping[proxy_name] = module.bias
                    module.track_running_stats = False
                    module.running_mean = None
                    module.running_var = None
        
        elif self.update_mode == 'last_layers':
            # 为最后几层和BN层创建代理参数
            for name, module in self.depth_model.named_modules():
                if isinstance(module, nn.Conv2d) and 'disp' in name:
                    if hasattr(module, 'weight') and module.weight is not None:
                        proxy_name = f"{name}.weight"
                        proxy_param = nn.Parameter(module.weight.data.clone())
                        self.proxy_params[proxy_name] = proxy_param
                        self.param_mapping[proxy_name] = module.weight
                    if hasattr(module, 'bias') and module.bias is not None:
                        proxy_name = f"{name}.bias"
                        proxy_param = nn.Parameter(module.bias.data.clone())
                        self.proxy_params[proxy_name] = proxy_param
                        self.param_mapping[proxy_name] = module.bias
                
                if isinstance(module, nn.BatchNorm2d):
                    if hasattr(module, 'weight') and module.weight is not None:
                        proxy_name = f"{name}.weight"
                        proxy_param = nn.Parameter(module.weight.data.clone())
                        self.proxy_params[proxy_name] = proxy_param
                        self.param_mapping[proxy_name] = module.weight
                    if hasattr(module, 'bias') and module.bias is not None:
                        proxy_name = f"{name}.bias"
                        proxy_param = nn.Parameter(module.bias.data.clone())
                        self.proxy_params[proxy_name] = proxy_param
                        self.param_mapping[proxy_name] = module.bias
                    module.track_running_stats = False
                    module.running_mean = None
                    module.running_var = None
    
    def apply_proxy_parameters(self):
        """临时将代理参数应用到模型中"""
        for proxy_name, proxy_param in self.proxy_params.items():
            if proxy_name in self.param_mapping:
                original_param = self.param_mapping[proxy_name]
                # 保存原始值并替换为代理参数
                if not hasattr(original_param, '_original_data'):
                    original_param._original_data = original_param.data.clone()
                original_param.data = proxy_param.data
    
    def restore_original_parameters(self):
        """恢复原始参数"""
        for proxy_name, original_param in self.param_mapping.items():
            if hasattr(original_param, '_original_data'):
                original_param.data = original_param._original_data
                delattr(original_param, '_original_data')
    
    def sample_virtual_poses(self, batch_size, device=None, num_poses=4, xyz_range=0.1, rpy_range=0.05):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        patterns = [
            [xyz_range, 0, 0, 0, rpy_range, 0],
            [-xyz_range, 0, 0, 0, -rpy_range, 0],
            [0, 0, xyz_range, rpy_range, 0, 0],
            [0, xyz_range, 0, 0, 0, rpy_range],
        ]
        poses_list = []
        for tx, ty, tz, rx, ry, rz in patterns:
            translation = torch.zeros(batch_size, 1, 1, 3, device=device)
            translation[:, 0, 0, 0] = tx
            translation[:, 0, 0, 1] = ty
            translation[:, 0, 0, 2] = tz
            axisangle = torch.zeros(batch_size, 1, 1, 3, device=device)
            axisangle[:, 0, 0, 0] = rx
            axisangle[:, 0, 0, 1] = ry
            axisangle[:, 0, 0, 2] = rz
            poses_list.append((axisangle, translation))
        return poses_list

    def back_project_depth(self, depth, K, pose):
        batch_size, _, height, width = depth.shape
        device = depth.device
        K_3x3 = K[:, :3, :3] if K.shape[-1] == 4 else K
        backproject = BackprojectDepth(batch_size, height, width).to(device)
        project = Project3D(batch_size, height, width).to(device)
        inv_K = torch.inverse(K_3x3)
        K_proj = torch.zeros(batch_size, 3, 4, device=device)
        K_proj[:, :3, :3] = K_3x3
        axisangle, translation = pose
        T = transformation_from_parameters(axisangle[:, 0], translation[:, 0], invert=True)
        cam_points = backproject(depth, inv_K)
        pix_coords = project(cam_points, K_proj, T)
        z = cam_points[:, 2, :].reshape(batch_size, 1, height, width)
        transformed_depth = F.grid_sample(z, pix_coords, mode='bilinear', padding_mode='zeros', align_corners=True)
        valid_mask = ((pix_coords[:, :, :, 0] >= -1) & (pix_coords[:, :, :, 0] <= 1) &
                      (pix_coords[:, :, :, 1] >= -1) & (pix_coords[:, :, :, 1] <= 1)).unsqueeze(1).float()
        return transformed_depth * valid_mask

    def create_visibility_mask(self, depth, back_depth, border_margin=2):
        valid = torch.isfinite(back_depth) & (back_depth > 1e-6) & torch.isfinite(depth) & (depth > 1e-6)
        mask = valid.float()
        if border_margin > 0:
            mask[:, :, :border_margin, :] = 0
            mask[:, :, -border_margin:, :] = 0
            mask[:, :, :, :border_margin] = 0
            mask[:, :, :, -border_margin:] = 0
        return mask

    def compute_scale_invariant_depth_error(self, depth, back_depth, mask):
        eps = 1e-6
        depth_safe = torch.clamp(depth, min=eps)
        back_depth_safe = torch.clamp(back_depth, min=eps)
        if mask is None:
            mask = torch.ones_like(depth_safe)
        valid = mask > 0.5
        if valid.sum() == 0:
            return torch.zeros((), device=depth.device)
        log_diff = (torch.log(depth_safe) - torch.log(back_depth_safe))[valid]
        n = log_diff.numel()
        loss = (log_diff.pow(2).sum() / n) - (log_diff.sum() ** 2) / (n ** 2)
        return loss

    def compute_gradient_consistency_loss(self, depth, back_depth, mask):
        if mask is None:
            mask = torch.ones_like(depth)
        def gradient_x(img):
            return img[:, :, :, 1:] - img[:, :, :, :-1]
        def gradient_y(img):
            return img[:, :, 1:, :] - img[:, :, :-1, :]
        gx1, gy1 = gradient_x(depth), gradient_y(depth)
        gx2, gy2 = gradient_x(back_depth), gradient_y(back_depth)
        mask_x = mask[:, :, :, 1:] * mask[:, :, :, :-1]
        mask_y = mask[:, :, 1:, :] * mask[:, :, :-1, :]
        diff_x = torch.abs(gx1 - gx2) * mask_x
        diff_y = torch.abs(gy1 - gy2) * mask_y
        denom = (mask_x.sum() + mask_y.sum()).clamp(min=1.0)
        return (diff_x.sum() + diff_y.sum()) / denom

    def forward(self, image, K, clean_depth=None):
        device = image.device
        batch_size = image.shape[0]
        self.depth_model.train()
        best_loss = float('inf')
        patience_counter = 0
        
        # 在第一次迭代前，确保原始参数被保存
        for proxy_name, original_param in self.param_mapping.items():
            if not hasattr(original_param, '_original_data'):
                original_param._original_data = original_param.data.clone()
        
        for step_idx in range(self.steps):
            # 应用代理参数（将更新后的代理参数应用到模型中）
            self.apply_proxy_parameters()
            
            # 需要梯度：深度必须可导，才能把几何一致性损失反传到代理参数
            depth = self.depth_model(image)
            if isinstance(depth, dict):
                depth = depth[("disp", 0)]
            poses = self.sample_virtual_poses(batch_size, device)
            total_loss = depth.new_tensor(0.0)
            for pose in poses:
                # 基于原视角深度，生成新视角"回投影"深度，避免重复图像渲染
                back_depth = self.back_project_depth(depth, K, pose)
                mask = self.create_visibility_mask(depth, back_depth)
                si_loss = self.compute_scale_invariant_depth_error(depth, back_depth, mask)
                grad_loss = self.compute_gradient_consistency_loss(depth, back_depth, mask)
                total_loss = total_loss + si_loss + 0.5 * grad_loss
            total_loss = total_loss / max(len(poses), 1)
            
            clean_loss = 0.0
            if clean_depth is not None:
                if clean_depth.dim() == 3:
                    clean_depth = clean_depth.unsqueeze(1)
                
                # 确保clean_depth和depth的形状匹配
                if clean_depth.shape != depth.shape:
                    # 调整clean_depth的形状以匹配depth
                    if clean_depth.shape[0] != depth.shape[0]:
                        min_batch_size = min(clean_depth.shape[0], depth.shape[0])
                        clean_depth = clean_depth[:min_batch_size]
                        depth = depth[:min_batch_size]
                    
                    # 调整空间维度
                    if clean_depth.shape[2:] != depth.shape[2:]:
                        clean_depth = F.interpolate(clean_depth, size=depth.shape[2:], mode='bilinear', align_corners=False)
                
                # 只有当clean_depth不为空时才计算损失
                if clean_depth.numel() > 0 and depth.numel() > 0:
                    clean_loss = F.l1_loss(depth, clean_depth)
                    total_loss = total_loss + 0.5 * clean_loss
            
            self.optimizer.zero_grad()
            total_loss.backward()
            if self.grad_clip is not None and len(self.params_to_update) > 0:
                torch.nn.utils.clip_grad_norm_(self.params_to_update, self.grad_clip)
            self.optimizer.step()
            # 注意：不在这里恢复原始参数，因为下一次迭代需要继续使用更新后的代理参数
            
            if self.early_stop:
                loss_val = float(total_loss.detach().item())
                if loss_val < best_loss:
                    best_loss = loss_val
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= self.early_stop_patience:
                    break
        
        # 最后一次应用代理参数进行推理
        self.apply_proxy_parameters()
        self.depth_model.eval()
        with torch.no_grad():
            final_depth = self.depth_model(image)
            if isinstance(final_depth, dict):
                final_depth = final_depth[("disp", 0)]
        # 恢复原始参数
        self.restore_original_parameters()
        return final_depth

# 3. EATA自适应模块，支持clean伪标签loss
class EATA(nn.Module):
    def __init__(self, model, optimizer, steps=1):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.criterion = nn.L1Loss()
        
    def forward(self, x, clean_depth=None):
        self.model.train()
        
        for step_idx in range(self.steps):
            out = self.model(x)
            if clean_depth is not None:
                if clean_depth.dim() == 3:
                    clean_depth = clean_depth.unsqueeze(1)
                # 确保形状匹配
                if clean_depth.shape != out.shape:
                    if clean_depth.shape[0] != out.shape[0]:
                        min_batch_size = min(clean_depth.shape[0], out.shape[0])
                        clean_depth = clean_depth[:min_batch_size]
                        out = out[:min_batch_size]
                    if clean_depth.shape[2:] != out.shape[2:]:
                        clean_depth = F.interpolate(clean_depth, size=out.shape[2:], mode='bilinear', align_corners=False)
                loss = self.criterion(out, clean_depth)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
        
        self.model.eval()
        with torch.no_grad():
            return self.model(x)

class RegressionSSA(nn.Module):
    """基于 regression-tta 的显著子空间对齐模块。"""
    def __init__(self, depth_model, stats_path, topk=64, steps=1, lr=1e-4,
                 update_mode="encoder_bn", feature_pool="avg", eps=1e-6,
                 weight_bias=1e-3, weight_exp=1.0, grad_clip=None):
        super().__init__()
        self.depth_model = depth_model
        self.steps = steps
        self.eps = eps
        self.grad_clip = grad_clip
        self.projector = FeatureProjector(pool_type=feature_pool)
        self.update_mode = update_mode

        mean, basis, var = load_pca_stats(stats_path, topk)
        self.register_buffer("ssa_mean", mean)
        self.register_buffer("ssa_basis", basis)
        self.register_buffer("ssa_var", var)
        self.register_buffer("dim_weight", build_dim_weight(var, weight_bias, weight_exp))

        self.params_to_update = self.configure_params_to_update()
        if not self.params_to_update:
            raise ValueError("SSA 未找到可训练参数，请检查 update_mode 设置。")
        self.optimizer = torch.optim.AdamW(self.params_to_update, lr=lr)

    def configure_params_to_update(self):
        params = []
        for p in self.depth_model.parameters():
            p.requires_grad = False

        if self.update_mode == "encoder_bn":
            for m in self.depth_model.encoder.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.requires_grad_(True)
                    m.track_running_stats = False
                    m.running_mean = None
                    m.running_var = None
                    params += list(m.parameters())
        elif self.update_mode == "encoder_all":
            for p in self.depth_model.encoder.parameters():
                p.requires_grad = True
                params.append(p)
        elif self.update_mode == "all":
            for p in self.depth_model.parameters():
                p.requires_grad = True
                params.append(p)
        else:
            raise ValueError(f"未知的 SSA 参数更新模式: {self.update_mode}")
        return params

    def compute_alignment_loss(self, features):
        f_pc = (features - self.ssa_mean) @ self.ssa_basis
        f_pc_mean = f_pc.mean(dim=0)
        f_pc_var = f_pc.var(dim=0, unbiased=False).clamp_min(self.eps)
        zeros = torch.zeros_like(f_pc_mean)
        kl_forward = diagonal_gaussian_kl_loss(
            f_pc_mean, f_pc_var + self.eps, zeros, self.ssa_var + self.eps, reduction="none")
        kl_reverse = diagonal_gaussian_kl_loss(
            zeros, self.ssa_var + self.eps, f_pc_mean, f_pc_var + self.eps, reduction="none")
        loss = (kl_forward + kl_reverse) * self.dim_weight
        return loss.sum()

    def forward(self, image, *_args, **_kwargs):
        self.depth_model.train()
        for _ in range(self.steps):
            feature_map = self.depth_model.encoder(image)
            feature_vec = self.projector(feature_map)
            loss = self.compute_alignment_loss(feature_vec)
            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.params_to_update, self.grad_clip)
            self.optimizer.step()
        self.depth_model.eval()
        with torch.no_grad():
            final_depth = self.depth_model(image)
            if isinstance(final_depth, dict):
                final_depth = final_depth[("disp", 0)]
        return final_depth

def configure_model_for_eata(model):
    for param in model.parameters():
        param.requires_grad = False
    if hasattr(model, 'decoder'):
        for p in model.decoder.parameters():
            p.requires_grad = True
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
            for p in m.parameters():
                p.requires_grad = True
    return model

def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)
    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

def main():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('--mode', type=str, choices=['generate_clean', 'adapt'], required=True, 
                       help='generate_clean:生成clean伪标签; adapt:自适应测试')
    parser.add_argument('--tta_mode', type=str, choices=['vec', 'vec_fast', 'eata', 'ssa'], default='vec', 
                       help='选择TTA模式: vec=VECTTA, vec_fast=VecTTAFast, eata=EATA, ssa=Regression SSA')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--log_dir', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='kitti')
    parser.add_argument('--eval_split', type=str, default='eigen')
    parser.add_argument('--backbone', type=str, default='resnet_lite')
    parser.add_argument('--height', type=int, default=192)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_layers', type=int, default=50)
    parser.add_argument('--num_features', type=int, default=256)
    parser.add_argument('--model_dim', type=int, default=32)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--dim_out', type=int, default=64)
    parser.add_argument('--query_nums', type=int, default=64)
    parser.add_argument('--min_depth', type=float, default=0.001)
    parser.add_argument('--max_depth', type=float, default=80.0)
    parser.add_argument('--load_weights_folder', type=str, required=True)
    parser.add_argument('--post_process', action='store_true')
    parser.add_argument('--disable_median_scaling', action='store_true')
    parser.add_argument('--pred_depth_scale_factor', type=float, default=1.0)
    parser.add_argument('--eata_steps', type=int, default=1)
    parser.add_argument('--clean_pred_path', type=str, default='clean_pred_disps.npy', 
                       help='clean伪标签保存/加载路径')
    
    # VECTTA 参数
    parser.add_argument('--vec_steps', type=int, default=5)
    parser.add_argument('--vec_update_mode', type=str, choices=['bn_only', 'bn_decoder', 'last_layers','all'], 
                       default='bn_only')
    parser.add_argument('--vec_lr', type=float, default=1e-4)
    parser.add_argument('--vec_early_stop', action='store_true')
    parser.add_argument('--vec_early_stop_patience', type=int, default=3)
    parser.add_argument('--vec_grad_clip', type=float, default=1.0)

    # SSA 参数
    parser.add_argument('--ssa_stats_path', type=str, help='SSA 特征统计文件路径')
    parser.add_argument('--ssa_topk', type=int, default=64)
    parser.add_argument('--ssa_steps', type=int, default=1)
    parser.add_argument('--ssa_lr', type=float, default=1e-4)
    parser.add_argument('--ssa_update_mode', type=str,
                        choices=['encoder_bn', 'encoder_all', 'all'], default='encoder_bn')
    parser.add_argument('--ssa_feature_pool', type=str, choices=['avg', 'max'], default='avg')
    parser.add_argument('--ssa_eps', type=float, default=1e-6)
    parser.add_argument('--ssa_weight_bias', type=float, default=1e-3)
    parser.add_argument('--ssa_weight_exp', type=float, default=1.0)
    parser.add_argument('--ssa_grad_clip', type=float, default=0.0)
    
    args = parser.parse_args()

    # === 数据集 ===
    splits_dir = os.path.join(os.path.dirname(__file__), "splits")
    filenames = readlines(os.path.join(splits_dir, args.eval_split, "test_files.txt"))
    from datasets.kitti_dataset import KITTIRAWDataset
    dataset = KITTIRAWDataset(
        args.data_path, filenames,
        args.height, args.width,
        [0], 1, is_train=False
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, 
                                            num_workers=4, pin_memory=True, drop_last=False)

    # === 加载encoder/decoder和权重 ===
    if args.backbone in ["resnet", "resnet_lite"]:
        encoder = networks.ResnetEncoderDecoder(
            num_layers=args.num_layers,
            num_features=args.num_features,
            model_dim=args.model_dim
        )
    elif args.backbone == "resnet18_lite":
        encoder = networks.LiteResnetEncoderDecoder(model_dim=args.model_dim)
    elif args.backbone == "eff_b5":
        encoder = networks.BaseEncoder.build(num_features=args.num_features, model_dim=args.model_dim)
    else:
        encoder = networks.Unet(pretrained=False, backbone=args.backbone, in_channels=3, 
                                num_classes=args.model_dim, decoder_channels=[1024, 512, 256, 128])

    encoder_path = os.path.join(args.load_weights_folder, "encoder.pth")
    encoder_dict = torch.load(encoder_path)
    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
    encoder.cuda().eval()

    if args.backbone.endswith("_lite"):
        decoder = networks.Lite_Depth_Decoder_QueryTr(
            in_channels=args.model_dim,
            patch_size=args.patch_size,
            dim_out=args.dim_out,
            embedding_dim=args.model_dim,
            query_nums=args.query_nums,
            num_heads=4,
            min_val=args.min_depth,
            max_val=args.max_depth
        )
    else:
        decoder = networks.Depth_Decoder_QueryTr(
            in_channels=args.model_dim,
            patch_size=args.patch_size,
            dim_out=args.dim_out,
            embedding_dim=args.model_dim,
            query_nums=args.query_nums,
            num_heads=4,
            min_val=args.min_depth,
            max_val=args.max_depth
        )
    decoder_path = os.path.join(args.load_weights_folder, "depth.pth")
    decoder.load_state_dict(torch.load(decoder_path))
    decoder.cuda().eval()
    
    model = SQLDepthModel(encoder, decoder)
    # 仅当使用EATA时解冻相应参数，VECTTA内部自行配置
    if args.tta_mode == 'eata':
        configure_model_for_eata(model)

    if args.mode == 'generate_clean':
        # === 生成clean伪标签 ===
        model.eval()
        pred_disps = []
        with torch.no_grad():
            for batch in dataloader:
                images = batch[("color", 0, 0)].cuda()
                pred_disp = model(images)
                pred_disp = pred_disp.cpu().numpy()
                if pred_disp.ndim == 4:
                    pred_disp = pred_disp[:, 0, :, :]
                pred_disps.append(pred_disp)
        pred_disps = np.concatenate(pred_disps, axis=0)
        np.save(args.clean_pred_path, pred_disps)
        print(f"Clean伪标签已保存到 {args.clean_pred_path}")
        return

    elif args.mode == 'adapt':
        # === 自适应推理 ===
        clean_pred_disps = None
        if args.tta_mode in {'vec', 'vec_fast', 'eata'}:
            clean_pred_disps = np.load(args.clean_pred_path)

        pred_disps = []
        idx = 0

        if args.tta_mode == 'vec':
            adapt_model = VECTTA(
                depth_model=model,
                update_mode=args.vec_update_mode,
                optimizer_lr=args.vec_lr,
                steps=args.vec_steps,
                early_stop=args.vec_early_stop,
                early_stop_patience=args.vec_early_stop_patience,
                grad_clip=args.vec_grad_clip,
            )
            adapt_model.cuda()
        elif args.tta_mode == 'vec_fast':
            adapt_model = VecTTAFast(
                depth_model=model,
                update_mode=args.vec_update_mode,
                optimizer_lr=args.vec_lr,
                steps=args.vec_steps,
                early_stop=args.vec_early_stop,
                early_stop_patience=args.vec_early_stop_patience,
                grad_clip=args.vec_grad_clip,
            )
            adapt_model.cuda()
        elif args.tta_mode == 'ssa':
            if not args.ssa_stats_path:
                raise ValueError("SSA 模式需要提供 --ssa_stats_path")
            adapt_model = RegressionSSA(
                depth_model=model,
                stats_path=args.ssa_stats_path,
                topk=args.ssa_topk,
                steps=args.ssa_steps,
                lr=args.ssa_lr,
                update_mode=args.ssa_update_mode,
                feature_pool=args.ssa_feature_pool,
                eps=args.ssa_eps,
                weight_bias=args.ssa_weight_bias,
                weight_exp=args.ssa_weight_exp,
                grad_clip=args.ssa_grad_clip if args.ssa_grad_clip > 0 else None,
            )
            adapt_model.cuda()
        else:
            optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=4e-5)
            adapt_model = EATA(model, optimizer, steps=args.eata_steps)

        for batch in dataloader:
            images = batch[("color", 0, 0)].cuda()
            K = batch[("K", 0)].cuda()
            clean_depth = None
            if clean_pred_disps is not None:
                clean_depth = torch.from_numpy(clean_pred_disps[idx:idx+images.shape[0]]).cuda()
                idx += images.shape[0]

            if args.tta_mode in ['vec', 'vec_fast']:
                pred_disp = adapt_model(images, K, clean_depth=clean_depth)
            elif args.tta_mode == 'ssa':
                pred_disp = adapt_model(images)
            else:
                pred_disp = adapt_model(images, clean_depth=clean_depth)

            pred_disp = pred_disp.cpu().numpy()
            if pred_disp.ndim == 4:
                pred_disp = pred_disp[:, 0, :, :]
            pred_disps.append(pred_disp)
            
        pred_disps = np.concatenate(pred_disps, axis=0)

        # === 评测 ===
        gt_path = os.path.join(splits_dir, args.eval_split, "gt_depths.npz")
        gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]
        errors = []
        ratios = []
        MIN_DEPTH = args.min_depth
        MAX_DEPTH = args.max_depth

        for i in range(pred_disps.shape[0]):
            gt_depth = gt_depths[i]
            gt_height, gt_width = gt_depth.shape[:2]
            pred_disp = pred_disps[i]
            pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
            pred_depth = pred_disp

            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

            pred_depth = pred_depth[mask]
            gt_depth = gt_depth[mask]

            pred_depth *= args.pred_depth_scale_factor
            if not args.disable_median_scaling:
                ratio = np.median(gt_depth) / np.median(pred_depth)
                ratios.append(ratio)
                pred_depth *= ratio

            pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
            pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

            errors.append(compute_errors(gt_depth, pred_depth))

        if not args.disable_median_scaling:
            ratios = np.array(ratios)
            med = np.median(ratios)
            print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

        mean_errors = np.array(errors).mean(0)
        print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
        print("\n-> Done!")

if __name__ == "__main__":
    main()


