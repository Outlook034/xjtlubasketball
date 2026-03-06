"""
VecTTA (Viewpoint Equivariance Consistency Test-Time Adaptation) for Depth Estimation on Cityscapes

本脚本基于 `cityscapes_ssaTTA.py`，将 SSA-TTA 替换为 VecTTA：
- 使用 tta_sqldepth_cityscapes_c.py 中的 VECTTA
- 数据集与评估流程参考 evaluate_cityscapes_vectta.py
- 模型结构与权重加载方式与 evaluate_cityscapes_vectta.py 保持一致
"""

import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm

from utils import readlines
from layers import disp_to_depth, BackprojectDepth, Project3D, transformation_from_parameters

# 使得可以导入上级目录的 datasets / networks
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import datasets
import networks


class SQLDepthModel(nn.Module):
    """SQLdepth 模型包装器"""
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, x):
        out = self.decoder(self.encoder(x))
        if isinstance(out, dict):
            return out[("disp", 0)]
        return out


class VECTTAProxyFast(nn.Module):
    """VecTTA Proxy-Fast变体：冻结所有模型参数，使用代理参数进行适应
    
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


class VECTTA(nn.Module):
    """几何等变一致测试时适应 (Viewpoint Equivariance Consistency Test-Time Adaptation)
    
    通过注入虚拟微小相机位姿变换，构建深度预测的几何闭环一致性约束。
    从 tta_sqldepth_cityscapes_c.py 复制
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


def compute_errors(gt, pred):
    """计算深度误差指标（与 evaluate_cityscapes_vectta.py 一致）"""
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)
    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def convert_arg_line_to_args(arg_line):
    """将参数文件中的每一行转换为参数列表（支持每行多个参数）"""
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield str(arg)


def main():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    # 设置参数文件解析函数（支持每行多个参数）
    parser.convert_arg_line_to_args = convert_arg_line_to_args
    
    # 数据与模型
    parser.add_argument('--data_path', type=str, default=None,
                        help='Cityscapes 数据根目录')
    parser.add_argument('--eval_data_path', type=str, default=None,
                        help='Cityscapes 数据根目录（与 data_path 等价，用于兼容参数文件）')
    parser.add_argument('--load_weights_folder', type=str, default=None,
                        help='模型权重文件夹路径')
    parser.add_argument('--eval_split', type=str, default='cityscapes')
    parser.add_argument('--height', type=int, default=192)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--model_dim', type=int, default=32)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--dim_out', type=int, default=128, help='与 MonodepthOptions 默认值一致')
    parser.add_argument('--query_nums', type=int, default=128, help='与 MonodepthOptions 默认值一致')
    parser.add_argument('--min_depth', type=float, default=1e-3)
    parser.add_argument('--max_depth', type=float, default=80.0)
    parser.add_argument('--disable_median_scaling', action='store_true')
    parser.add_argument('--pred_depth_scale_factor', type=float, default=1.0)
    
    # 兼容参数文件中的其他参数（即使不使用也允许存在）
    parser.add_argument('--dataset', type=str, default=None, help='数据集名称（兼容参数文件，未使用）')
    parser.add_argument('--split', type=str, default=None, help='Split 名称（兼容参数文件，未使用）')
    parser.add_argument('--eval_mono', action='store_true', help='单目评估（兼容参数文件，未使用）')
    parser.add_argument('--eval_stereo', action='store_true', help='双目评估（兼容参数文件，未使用）')
    parser.add_argument('--post_process', action='store_true', help='后处理（兼容参数文件，未使用）')
    parser.add_argument('--save_pred_disps', action='store_true', help='保存预测视差（兼容参数文件，未使用）')

    # VecTTA 参数（参考 tta_sqldepth_cityscapes_c.py）
    parser.add_argument('--vec_steps', type=int, default=5, help='VecTTA 适应步数')
    parser.add_argument('--vec_lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--vec_update_mode', type=str, 
                        choices=['bn_only', 'bn_decoder', 'last_layers', 'all'], default='bn_only',
                        help='更新模式：bn_only=仅BatchNorm, bn_decoder=BatchNorm+Decoder, all=全部参数')
    parser.add_argument('--vec_early_stop', action='store_true', help='启用早停机制')
    parser.add_argument('--vec_early_stop_patience', type=int, default=3, help='早停耐心值')
    parser.add_argument('--vec_grad_clip', type=float, default=1.0, help='梯度裁剪阈值')
    parser.add_argument('--use_proxy_fast', action='store_true', 
                       help='使用 Proxy-fast 变体：冻结所有模型参数，使用代理参数进行适应')
    parser.add_argument('--clean_pred_path', type=str, default=None,
                       help='clean 伪标签路径（可选，用于额外的监督）')
    parser.add_argument('--generate_clean', action='store_true',
                       help='生成 clean 伪标签（不使用 TTA），保存后退出')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader worker 数量')
    parser.add_argument('--disable_tta', action='store_true', help='禁用 VecTTA，直接使用原始模型预测（用于对比）')

    args = parser.parse_args()
    
    # 统一处理 data_path 和 eval_data_path（兼容参数文件）
    if args.eval_data_path is not None:
        args.data_path = args.eval_data_path
    if args.data_path is None:
        raise ValueError("必须提供 --data_path 或 --eval_data_path 参数")
    
    # 检查并处理 load_weights_folder
    if args.load_weights_folder is None:
        raise ValueError("必须提供 --load_weights_folder 参数（可在参数文件中指定）")
    # 确保 load_weights_folder 被展开（处理相对路径和 ~）
    args.load_weights_folder = os.path.expanduser(args.load_weights_folder)
    # 如果是相对路径，转换为绝对路径（相对于脚本所在目录）
    if not os.path.isabs(args.load_weights_folder):
        args.load_weights_folder = os.path.abspath(args.load_weights_folder)

    # 数据集与文件列表
    splits_dir = os.path.join(os.path.dirname(__file__), "splits")
    test_files_path = os.path.join(splits_dir, args.eval_split, "test_files.txt")
    if not os.path.exists(test_files_path):
        raise FileNotFoundError(
            f"Test files not found: {test_files_path}\n"
            f"Please check your --eval_split argument."
        )

    filenames = readlines(test_files_path)

    # Cityscapes 数据集
    frames_to_load = [0]
    encoder_path = os.path.join(args.load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(args.load_weights_folder, "depth.pth")

    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Encoder weights not found: {encoder_path}")
    if not os.path.exists(decoder_path):
        raise FileNotFoundError(f"Decoder weights not found: {decoder_path}")

    encoder_dict = torch.load(encoder_path)

    try:
        HEIGHT, WIDTH = encoder_dict['height'], encoder_dict['width']
    except KeyError:
        print('No "height" or "width" keys found in the encoder state_dict, '
              'resorting to using command line values!')
        HEIGHT, WIDTH = args.height, args.width

    dataset = datasets.CityscapesEvalDataset(
        args.data_path,
        filenames,
        HEIGHT,
        WIDTH,
        frames_to_load,
        4,
        is_train=False,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # 构建模型（与 evaluate_cityscapes_vectta.py 一致）
    encoder = networks.Resnet50EncoderDecoder(model_dim=args.model_dim)
    
    # 先加载 decoder checkpoint 以推断正确的参数（避免 size mismatch）
    decoder_dict = torch.load(decoder_path)
    
    # 从 checkpoint 中推断 dim_out 和 query_nums
    # 检查 convert_to_prob.0.weight 的形状来推断 dim_out
    if 'convert_to_prob.0.weight' in decoder_dict:
        # convert_to_prob.0.weight 形状是 [dim_out, query_nums, 1, 1]
        dim_out_from_checkpoint = decoder_dict['convert_to_prob.0.weight'].shape[0]
        query_nums_from_checkpoint = decoder_dict['convert_to_prob.0.weight'].shape[1]
        print(f"-> Inferring dim_out={dim_out_from_checkpoint}, query_nums={query_nums_from_checkpoint} from checkpoint")
        # 使用 checkpoint 中的值，如果命令行参数不同则警告
        if args.dim_out != dim_out_from_checkpoint:
            print(f"   Warning: dim_out mismatch! Using {dim_out_from_checkpoint} from checkpoint (args: {args.dim_out})")
        if args.query_nums != query_nums_from_checkpoint:
            print(f"   Warning: query_nums mismatch! Using {query_nums_from_checkpoint} from checkpoint (args: {args.query_nums})")
        actual_dim_out = dim_out_from_checkpoint
        actual_query_nums = query_nums_from_checkpoint
    else:
        # 如果找不到，使用命令行参数
        print(f"-> Using dim_out={args.dim_out}, query_nums={args.query_nums} from command line")
        actual_dim_out = args.dim_out
        actual_query_nums = args.query_nums
    
    depth_decoder = networks.Depth_Decoder_QueryTr(
        in_channels=args.model_dim,
        patch_size=args.patch_size,
        dim_out=actual_dim_out,
        embedding_dim=args.model_dim,
        query_nums=actual_query_nums,
        num_heads=4,
        min_val=args.min_depth,
        max_val=args.max_depth,
    )

    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
    depth_decoder.load_state_dict(decoder_dict, strict=True)

    encoder.cuda()
    depth_decoder.cuda()

    depth_model = SQLDepthModel(encoder, depth_decoder)

    # Generate clean predictions mode
    if args.generate_clean:
        print("-> Generating clean predictions (no TTA)")
        depth_model.eval()
        
        pred_disps = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Generating clean predictions"):
                images = batch[("color", 0, 0)].cuda()
                
                # Normal inference without TTA
                if images.max() > 1.0:
                    images = images / 255.0
                mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
                std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
                images_norm = (images - mean) / std
                
                output = depth_decoder(encoder(images_norm))
                pred_disp = output[("disp", 0)]  # Get disparity directly
                pred_disp = pred_disp.cpu().numpy()
                if pred_disp.ndim == 4:
                    pred_disp = pred_disp[:, 0, :, :]  # [B, H, W]
                pred_disps.append(pred_disp)
        
        pred_disps = np.concatenate(pred_disps, axis=0)
        clean_pred_path = args.clean_pred_path if args.clean_pred_path else 'clean_pred_disps_cityscapes.npy'
        np.save(clean_pred_path, pred_disps)
        print(f"-> Clean predictions saved to {clean_pred_path}")
        print(f"-> Shape: {pred_disps.shape}")
        return

    # 加载 clean predictions（如果提供）
    clean_pred_disps = None
    if args.clean_pred_path and os.path.exists(args.clean_pred_path):
        clean_pred_disps = np.load(args.clean_pred_path)
        print(f"-> Loaded clean predictions from {args.clean_pred_path}")

    # 创建 VecTTA 适配器
    if not args.disable_tta:
        if args.use_proxy_fast:
            vectta = VECTTAProxyFast(
                depth_model=depth_model,
                update_mode=args.vec_update_mode,
                optimizer_lr=args.vec_lr,
                steps=args.vec_steps,
                early_stop=args.vec_early_stop,
                early_stop_patience=args.vec_early_stop_patience,
                grad_clip=args.vec_grad_clip,
            ).cuda()
            print("-> VecTTA Proxy-Fast enabled with update_mode={}, steps={}, lr={}".format(
                args.vec_update_mode, args.vec_steps, args.vec_lr))
        else:
            vectta = VECTTA(
                depth_model=depth_model,
                update_mode=args.vec_update_mode,
                optimizer_lr=args.vec_lr,
                steps=args.vec_steps,
                early_stop=args.vec_early_stop,
                early_stop_patience=args.vec_early_stop_patience,
                grad_clip=args.vec_grad_clip,
            ).cuda()
            print("-> VecTTA enabled with update_mode={}, steps={}, lr={}".format(
                args.vec_update_mode, args.vec_steps, args.vec_lr))
    else:
        vectta = None
        depth_model.eval()
        print("-> VecTTA disabled, using original model")

    # 推理
    pred_depths = []
    idx = 0

    print("-> Computing predictions with size {}x{}".format(HEIGHT, WIDTH))
    for batch in tqdm(dataloader, desc="Evaluating Cityscapes with VecTTA"):
        images = batch[("color", 0, 0)].cuda().float()
        bs = images.shape[0]

        if args.disable_tta:
            # 直接使用原始模型预测（用于对比）
            with torch.no_grad():
                if images.max() > 1.0:
                    images = images / 255.0
                mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
                std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
                images_norm = (images - mean) / std

                output = depth_decoder(encoder(images_norm))
                pred_disp, _ = disp_to_depth(output[("disp", 0)], args.min_depth, args.max_depth)
                pred_disp = pred_disp.cpu()[:, 0].numpy()
        else:
            # 获取 K 矩阵（相机内参）
            K = None
            if ('K', 0) in batch:
                K = batch[('K', 0)]
                if torch.cuda.is_available():
                    K = K.cuda()
            else:
                # 如果 K 不在数据中，创建默认的 K 矩阵（Cityscapes 归一化内参）
                batch_size = images.shape[0]
                fx = fy = 0.58  # 近似归一化焦距
                cx = cy = 0.5
                K = torch.zeros(batch_size, 4, 4, device=images.device)
                K[:, 0, 0] = fx
                K[:, 1, 1] = fy
                K[:, 0, 2] = cx
                K[:, 1, 2] = cy
                K[:, 2, 2] = 1.0
                K[:, 3, 3] = 1.0
                if idx == 0:  # 只打印一次警告
                    print("Warning: K matrix not found in data, using default Cityscapes intrinsics")
            
            # 准备 clean_depth（如果提供）
            clean_depth = None
            if clean_pred_disps is not None:
                clean_disp_batch = clean_pred_disps[idx:idx+bs]
                # 确保形状是 [B, H, W] 或 [B, 1, H, W]
                if clean_disp_batch.ndim == 2:
                    clean_disp_batch = clean_disp_batch[np.newaxis, ...]
                if clean_disp_batch.ndim == 3:
                    clean_disp_batch = clean_disp_batch[:, np.newaxis, ...]
                clean_depth = torch.from_numpy(clean_disp_batch).float().cuda()
                idx += bs
            
            # VecTTA 适应和预测
            # 注意：VecTTA 需要输入归一化的图像
            if images.max() > 1.0:
                images = images / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
            images_norm = (images - mean) / std
            
            # VecTTA forward pass（返回视差）
            pred_disp = vectta(images_norm, K, clean_depth=clean_depth)
            # pred_disp shape is [B, 1, H, W], VecTTA returns disparity
            # Convert using disp_to_depth to match original code format
            pred_disp, _ = disp_to_depth(pred_disp, args.min_depth, args.max_depth)
            if pred_disp.dim() == 4:
                pred_disp = pred_disp[:, 0, :, :]
            pred_disp = pred_disp.cpu().numpy()

        pred_depths.append(pred_disp)

    pred_depths = np.concatenate(pred_depths, axis=0)

    # 评估（Cityscapes 版本，参考 evaluate_cityscapes_vectta.py）
    gt_depths_dir = os.path.join(splits_dir, args.eval_split, "gt_depths")
    if not os.path.isdir(gt_depths_dir):
        raise FileNotFoundError(f"Cityscapes gt_depths directory not found: {gt_depths_dir}")

    errors = []
    ratios = []
    MIN_DEPTH = args.min_depth
    MAX_DEPTH = args.max_depth

    print("-> Evaluating on Cityscapes")
    for i in tqdm(range(pred_depths.shape[0]), desc="Computing metrics"):
        gt_path = os.path.join(gt_depths_dir, f"{i:03d}_depth.npy")
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"GT depth not found: {gt_path}")
        gt_depth = np.load(gt_path)
        gt_height, gt_width = gt_depth.shape[:2]

        # Cityscapes: 先去掉底部 25%（与 evaluate_cityscapes_vectta 中一致）
        gt_height = int(round(gt_height * 0.75))
        gt_depth = gt_depth[:gt_height]

        pred_disp = np.squeeze(pred_depths[i])
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = pred_disp
        # pred_depth = 1 / pred_disp  # 注释掉，因为 disp_to_depth 已经转换了

        # Cityscapes: 再做中心裁剪（中间 50%）
        gt_depth = gt_depth[256:, 192:1856]
        pred_depth = pred_depth[256:, 192:1856]

        mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

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
    print("\nVecTTA on SQLDepth (Cityscapes)")
    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")


if __name__ == "__main__":
    main()

