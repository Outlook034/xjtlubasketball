"""
CoTTA (Continual Test-Time Adaptation) for Depth Estimation
将 CoTTA 从分类任务适配到 SQLdepth 深度估计任务

核心机制：
1. EMA (Exponential Moving Average) 更新 teacher 模型
2. 使用 anchor 模型评估预测置信度（基于深度预测的稳定性）
3. 低置信度时使用数据增强进行多次预测并平均
4. 使用 scale-invariant loss 替代 softmax entropy
5. 随机恢复机制（stochastic restore）
"""

import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from copy import deepcopy
from PIL import Image
import argparse
from tqdm import tqdm

from layers import disp_to_depth
from utils import readlines
import networks


class DepthAugmentation:
    """针对深度估计的数据增强"""
    def __init__(self, gaussian_std=0.005, soft=False):
        self.gaussian_std = gaussian_std
        self.soft = soft
        
    def __call__(self, img):
        """
        Args:
            img: torch.Tensor [C, H, W] in range [0, 1]
        Returns:
            augmented_img: torch.Tensor [C, H, W]
        """
        # 确保输入是 torch.Tensor 且在 [0, 1] 范围
        if not isinstance(img, torch.Tensor):
            if isinstance(img, Image.Image):
                img = transforms.ToTensor()(img)
            else:
                img = torch.tensor(img)
        
        if img.max() > 1.0:
            img = img / 255.0
        
        if img.dim() == 4:
            img = img.squeeze(0)
        
        # 确保是 [C, H, W] 格式
        if img.dim() != 3:
            raise ValueError(f"Expected 3D tensor [C, H, W], got {img.shape}")
        
        # 颜色抖动
        if self.soft:
            brightness = np.random.uniform(0.8, 1.2)
            contrast = np.random.uniform(0.85, 1.15)
            saturation = np.random.uniform(0.75, 1.25)
            hue = np.random.uniform(-0.03, 0.03)
            gamma = np.random.uniform(0.85, 1.15)
        else:
            brightness = np.random.uniform(0.6, 1.4)
            contrast = np.random.uniform(0.7, 1.3)
            saturation = np.random.uniform(0.5, 1.5)
            hue = np.random.uniform(-0.06, 0.06)
            gamma = np.random.uniform(0.7, 1.3)
        
        # 应用颜色变换
        img = transforms.functional.adjust_brightness(img, brightness)
        img = transforms.functional.adjust_contrast(img, contrast)
        img = transforms.functional.adjust_saturation(img, saturation)
        img = transforms.functional.adjust_hue(img, hue)
        img = transforms.functional.adjust_gamma(img, gamma)
        
        # 几何变换
        h, w = img.shape[1], img.shape[2]
        padding = int(min(h, w) / 2)
        img = F.pad(img.unsqueeze(0), (padding, padding, padding, padding), mode='replicate')
        img = img.squeeze(0)
        
        # 随机仿射变换
        if self.soft:
            angle = np.random.uniform(-8, 8)
            translate = (np.random.uniform(-w/16, w/16), np.random.uniform(-h/16, h/16))
            scale = np.random.uniform(0.95, 1.05)
        else:
            angle = np.random.uniform(-15, 15)
            translate = (np.random.uniform(-w/16, w/16), np.random.uniform(-h/16, h/16))
            scale = np.random.uniform(0.9, 1.1)
        
        img = transforms.functional.affine(
            img, angle=angle, translate=translate, scale=scale, shear=0, 
            interpolation=transforms.InterpolationMode.BILINEAR
        )
        
        # 高斯模糊
        if self.soft:
            sigma = np.random.uniform(0.001, 0.25)
        else:
            sigma = np.random.uniform(0.001, 0.5)
        img = transforms.functional.gaussian_blur(img, kernel_size=5, sigma=sigma)
        
        # 中心裁剪回原尺寸
        img = transforms.functional.center_crop(img, (h, w))
        
        # 随机水平翻转
        if np.random.rand() < 0.5:
            img = transforms.functional.hflip(img)
        
        # 高斯噪声
        noise = torch.randn_like(img) * self.gaussian_std
        img = img + noise
        
        # 裁剪到 [0, 1]
        img = torch.clamp(img, 0.0, 1.0)
        
        return img


def get_depth_tta_transforms(gaussian_std=0.005, soft=False):
    """获取深度估计的 TTA 数据增强"""
    return DepthAugmentation(gaussian_std=gaussian_std, soft=soft)


def update_ema_variables(ema_model, model, alpha_teacher):
    """更新 EMA 模型参数"""
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model


def scale_invariant_loss(pred, target, mask=None):
    """
    计算 scale-invariant loss（用于深度估计）
    
    Args:
        pred: 预测深度 [B, 1, H, W] 或 [B, H, W]
        target: 目标深度 [B, 1, H, W] 或 [B, H, W]
        mask: 有效掩码 [B, 1, H, W] 或 [B, H, W] (可选)
    
    Returns:
        loss: scale-invariant loss
    """
    # 确保维度一致
    if pred.dim() == 4:
        pred = pred.squeeze(1)
    if target.dim() == 4:
        target = target.squeeze(1)
    if mask is not None and mask.dim() == 4:
        mask = mask.squeeze(1)
    
    # 计算 log 差值
    eps = 1e-6
    pred_log = torch.log(torch.clamp(pred, min=eps))
    target_log = torch.log(torch.clamp(target, min=eps))
    log_diff = pred_log - target_log
    
    if mask is not None:
        log_diff = log_diff * mask
        n = mask.sum()
        if n == 0:
            return torch.tensor(0.0, device=pred.device)
    else:
        n = log_diff.numel()
    
    # Scale-invariant loss: variance of log differences
    variance_focus = 0.85
    loss = torch.sqrt((log_diff ** 2).sum() / n - variance_focus * (log_diff.sum() / n) ** 2)
    
    return loss


def compute_depth_confidence(depth_pred, anchor_depth, threshold=0.1):
    """
    计算深度预测的置信度（基于与 anchor 模型预测的一致性）
    
    Args:
        depth_pred: 当前模型预测的深度 [B, 1, H, W] 或 [B, H, W]
        anchor_depth: anchor 模型预测的深度 [B, 1, H, W] 或 [B, H, W]
        threshold: 一致性阈值
    
    Returns:
        confidence: 置信度值 [B] 或标量
    """
    if depth_pred.dim() == 4:
        depth_pred = depth_pred.squeeze(1)
    if anchor_depth.dim() == 4:
        anchor_depth = anchor_depth.squeeze(1)
    
    # 计算相对误差
    eps = 1e-6
    relative_error = torch.abs(depth_pred - anchor_depth) / (anchor_depth + eps)
    
    # 一致性掩码（误差小于阈值）
    consistent_mask = relative_error < threshold
    
    # 置信度为一致像素的比例
    confidence = consistent_mask.float().mean(dim=(1, 2))  # [B]
    
    return confidence.mean()  # 返回平均置信度


class SQLDepthModel(nn.Module):
    """SQLdepth 模型包装器"""
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, x):
        features = self.encoder(x)
        out = self.decoder(features)
        if isinstance(out, dict):
            return out[("disp", 0)]
        return out


class CoTTADepth(nn.Module):
    """
    CoTTA for Depth Estimation
    
    将 CoTTA 适配到深度估计任务：
    - 使用 scale-invariant loss 替代 softmax entropy
    - 使用深度预测的置信度替代分类置信度
    - 适配数据增强到深度估计任务
    """
    def __init__(self, model, optimizer, steps=1, episodic=False, 
                 mt_alpha=0.999, rst_m=0.001, ap=0.9, num_aug=32,
                 min_depth=0.1, max_depth=80.0):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "cotta requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        
        # 保存模型和优化器状态
        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            self.copy_model_and_optimizer(self.model, self.optimizer)
        
        # 数据增强
        self.transform = get_depth_tta_transforms(soft=False)
        
        # 超参数
        self.mt = mt_alpha  # EMA 更新率
        self.rst = rst_m    # 随机恢复概率
        self.ap = ap        # 置信度阈值
        self.num_aug = num_aug  # 增强次数
        
        self.min_depth = min_depth
        self.max_depth = max_depth
        
        # 图像归一化参数
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    def copy_model_and_optimizer(self, model, optimizer):
        """复制模型和优化器状态"""
        model_state = deepcopy(model.state_dict())
        model_anchor = deepcopy(model)
        optimizer_state = deepcopy(optimizer.state_dict())
        ema_model = deepcopy(model)
        for param in ema_model.parameters():
            param.detach_()
        return model_state, optimizer_state, ema_model, model_anchor
    
    def load_model_and_optimizer(self, model, optimizer, model_state, optimizer_state):
        """恢复模型和优化器状态"""
        model.load_state_dict(model_state, strict=True)
        optimizer.load_state_dict(optimizer_state)
    
    def forward(self, x):
        """前向传播，返回深度（SQLdepth 输出的是深度值，虽然键名是 "disp"）"""
        if self.episodic:
            self.reset()
        
        # 归一化输入
        if x.max() > 1.0:
            x = x / 255.0
        device = x.device
        if self.mean.device != device:
            self.mean = self.mean.to(device)
            self.std = self.std.to(device)
        x_norm = (x - self.mean) / self.std
        
        for _ in range(self.steps):
            self.forward_and_adapt(x_norm, self.model, self.optimizer)
        
        # 返回最终预测的深度（SQLdepth 输出的是深度值）
        with torch.no_grad():
            final_output = self.model(x_norm)
            if isinstance(final_output, dict):
                final_depth = final_output[("disp", 0)]
            else:
                final_depth = final_output
        
        return final_depth
    
    def reset(self):
        """重置模型到初始状态"""
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        self.load_model_and_optimizer(self.model, self.optimizer,
                                     self.model_state, self.optimizer_state)
        # 同时重置 teacher 模型
        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            self.copy_model_and_optimizer(self.model, self.optimizer)
    
    @torch.enable_grad()
    def forward_and_adapt(self, x, model, optimizer):
        """
        前向传播并适应
        
        Args:
            x: 归一化后的输入图像 [B, 3, H, W]
            model: 学生模型
            optimizer: 优化器
        """
        # 学生模型预测
        # 注意：SQLdepth 的输出键名是 "disp"，但实际输出的是深度值
        student_output = model(x)
        if isinstance(student_output, dict):
            student_depth = student_output[("disp", 0)]
        else:
            student_depth = student_output
        
        # Anchor 模型预测（用于置信度评估）
        with torch.no_grad():
            anchor_output = self.model_anchor(x)
            if isinstance(anchor_output, dict):
                anchor_depth = anchor_output[("disp", 0)]
            else:
                anchor_depth = anchor_output
        
        # 计算置信度
        confidence = compute_depth_confidence(student_depth, anchor_depth)
        
        # Teacher (EMA) 模型预测
        with torch.no_grad():
            ema_output = self.model_ema(x)
            if isinstance(ema_output, dict):
                ema_depth = ema_output[("disp", 0)]
            else:
                ema_depth = ema_output
        
        # 如果置信度低，使用数据增强进行多次预测并平均
        if confidence < self.ap:
            # 将归一化图像转换回 [0, 1] 范围进行增强
            x_unnorm = x * self.std + self.mean
            x_unnorm = torch.clamp(x_unnorm, 0.0, 1.0)
            
            aug_depths = []
            for _ in range(self.num_aug):
                # 对批次中的每个样本分别进行增强
                x_aug_list = []
                for b in range(x_unnorm.shape[0]):
                    x_b = x_unnorm[b]  # [C, H, W]
                    x_aug_b = self.transform(x_b)
                    x_aug_list.append(x_aug_b)
                x_aug = torch.stack(x_aug_list, dim=0)  # [B, C, H, W]
                
                # 重新归一化
                x_aug_norm = (x_aug - self.mean) / self.std
                
                with torch.no_grad():
                    aug_output = self.model_ema(x_aug_norm)
                    if isinstance(aug_output, dict):
                        aug_depth = aug_output[("disp", 0)]
                    else:
                        aug_depth = aug_output
                    aug_depths.append(aug_depth)
            
            # 平均增强预测
            if len(aug_depths) > 0:
                ema_depth = torch.stack(aug_depths).mean(0)
        
        # 计算损失（scale-invariant loss）
        loss = scale_invariant_loss(student_depth, ema_depth.detach())
        
        # 反向传播和更新
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # 更新 EMA teacher 模型
        self.model_ema = update_ema_variables(
            ema_model=self.model_ema, 
            model=self.model, 
            alpha_teacher=self.mt
        )
        
        # 随机恢复（stochastic restore）
        for nm, m in self.model.named_modules():
            for npp, p in m.named_parameters():
                if npp in ['weight', 'bias'] and p.requires_grad:
                    mask = (torch.rand(p.shape, device=p.device) < self.rst).float()
                    with torch.no_grad():
                        state_key = f"{nm}.{npp}"
                        if state_key in self.model_state:
                            p.data = self.model_state[state_key] * mask + p * (1. - mask)
        
        # 不返回任何值，因为 forward 方法会单独获取最终预测
        return None


def collect_params(model):
    """收集所有可训练参数"""
    params = []
    names = []
    for nm, m in model.named_modules():
        for np, p in m.named_parameters():
            if np in ['weight', 'bias'] and p.requires_grad:
                params.append(p)
                names.append(f"{nm}.{np}")
    return params, names


def configure_model(model):
    """配置模型用于 CoTTA"""
    # 训练模式
    model.train()
    # 先禁用所有梯度
    model.requires_grad_(False)
    # 启用需要更新的参数
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        else:
            m.requires_grad_(True)
    return model


def compute_errors(gt, pred):
    """计算深度误差指标"""
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


def main():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--load_weights_folder', type=str, required=True)
    parser.add_argument('--eval_split', type=str, default='eigen')
    parser.add_argument('--backbone', type=str, default='resnet_lite')
    parser.add_argument('--height', type=int, default=192)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=50)
    parser.add_argument('--num_features', type=int, default=256)
    parser.add_argument('--model_dim', type=int, default=32)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--dim_out', type=int, default=64)
    parser.add_argument('--query_nums', type=int, default=64)
    parser.add_argument('--min_depth', type=float, default=0.1)
    parser.add_argument('--max_depth', type=float, default=80.0)
    parser.add_argument('--disable_median_scaling', action='store_true')
    parser.add_argument('--pred_depth_scale_factor', type=float, default=1.0)
    
    # CoTTA 参数
    parser.add_argument('--cotta_steps', type=int, default=1, help='CoTTA 适应步数')
    parser.add_argument('--cotta_episodic', action='store_true', help='是否使用 episodic 模式')
    parser.add_argument('--mt_alpha', type=float, default=0.999, help='EMA 更新率')
    parser.add_argument('--rst_m', type=float, default=0.001, help='随机恢复概率')
    parser.add_argument('--ap', type=float, default=0.9, help='置信度阈值')
    parser.add_argument('--num_aug', type=int, default=32, help='数据增强次数')
    parser.add_argument('--lr', type=float, default=1e-5, help='学习率（建议较小值，如1e-5）')
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader worker数量（设为0便于调试路径问题）')
    parser.add_argument('--disable_cotta', action='store_true', help='禁用CoTTA，直接使用原始模型预测（用于对比）')
    
    args = parser.parse_args()
    
    # 验证数据路径
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(
            f"Data path does not exist: {args.data_path}\n"
            f"Please check your --data_path argument. "
            f"It should point to the KITTI dataset root directory."
        )
    
    # 数据集
    splits_dir = os.path.join(os.path.dirname(__file__), "splits")
    test_files_path = os.path.join(splits_dir, args.eval_split, "test_files.txt")
    if not os.path.exists(test_files_path):
        raise FileNotFoundError(
            f"Test files not found: {test_files_path}\n"
            f"Please check your --eval_split argument."
        )
    
    filenames = readlines(test_files_path)
    
    # 验证第一个文件是否存在（用于快速检查路径配置）
    if len(filenames) > 0:
        first_line = filenames[0].split()
        if len(first_line) >= 2:
            folder = first_line[0]
            frame_index = int(first_line[1])
            side = first_line[2] if len(first_line) >= 3 else "l"
            
            # 构建示例路径
            from datasets.kitti_dataset import KITTIRAWDataset
            temp_dataset = KITTIRAWDataset(
                args.data_path, [filenames[0]],
                args.height, args.width,
                [0], 1, is_train=False
            )
            sample_path = temp_dataset.get_image_path(folder, frame_index, side)
            
            if not os.path.exists(sample_path):
                print(f"\n{'='*60}")
                print(f"Warning: Sample image not found!")
                print(f"{'='*60}")
                print(f"Data path: {args.data_path}")
                print(f"Folder from test file: {folder}")
                print(f"Frame index: {frame_index}")
                print(f"Side: {side}")
                print(f"Expected image path: {sample_path}")
                print(f"\nTrying to find alternative paths...")
                
                # 尝试查找可能的路径
                possible_paths = [
                    sample_path,
                    os.path.join(args.data_path, folder, f"image_0{temp_dataset.side_map[side]}/data/{frame_index:010d}.png"),
                    os.path.join(args.data_path, folder, f"image_0{temp_dataset.side_map[side]}/data/{frame_index:010d}.jpg"),
                ]
                
                # 检查父目录是否存在
                parent_dir = os.path.dirname(sample_path)
                if os.path.exists(parent_dir):
                    print(f"Parent directory exists: {parent_dir}")
                    # 列出目录中的文件（最多5个）
                    try:
                        files = os.listdir(parent_dir)
                        if files:
                            print(f"Files in directory (showing first 5):")
                            for f in files[:5]:
                                print(f"  - {f}")
                    except:
                        pass
                else:
                    print(f"Parent directory does not exist: {parent_dir}")
                    
                    # 尝试查找可能的父目录
                    possible_parents = [
                        parent_dir,
                        os.path.join(args.data_path, folder),
                        os.path.join(args.data_path, folder, f"image_0{temp_dataset.side_map[side]}"),
                    ]
                    for pp in possible_parents:
                        if os.path.exists(pp):
                            print(f"Found existing directory: {pp}")
                            break
                
                print(f"\n{'='*60}")
                print(f"Please check:")
                print(f"1. The --data_path should point to KITTI dataset root")
                print(f"2. Expected structure: <data_path>/{folder}/image_0X/data/XXXXX.png")
                print(f"3. Verify the folder name in test_files.txt matches your dataset")
                print(f"4. Check if image files exist in the expected location")
                print(f"{'='*60}\n")
                
                # 不抛出异常，让用户看到详细信息后决定是否继续
                # raise FileNotFoundError(f"Sample image not found: {sample_path}")
    
    from datasets.kitti_dataset import KITTIRAWDataset
    dataset = KITTIRAWDataset(
        args.data_path, filenames,
        args.height, args.width,
        [0], 1, is_train=False
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False
    )
    
    # 加载模型
    if args.backbone in ["resnet", "resnet_lite"]:
        encoder = networks.ResnetEncoderDecoder(
            num_layers=args.num_layers,
            num_features=args.num_features,
            model_dim=args.model_dim
        )
    elif args.backbone == "resnet18_lite":
        encoder = networks.LiteResnetEncoderDecoder(model_dim=args.model_dim)
    elif args.backbone == "eff_b5":
        encoder = networks.BaseEncoder.build(
            num_features=args.num_features, 
            model_dim=args.model_dim
        )
    else:
        encoder = networks.Unet(
            pretrained=False, backbone=args.backbone, in_channels=3,
            num_classes=args.model_dim, 
            decoder_channels=[1024, 512, 256, 128]
        )
    
    encoder_path = os.path.join(args.load_weights_folder, "encoder.pth")
    encoder_dict = torch.load(encoder_path)
    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
    encoder.cuda()
    
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
    decoder.cuda()
    
    model = SQLDepthModel(encoder, decoder)
    
    # 配置模型
    model = configure_model(model)
    
    # 收集参数并创建优化器
    params, names = collect_params(model)
    optimizer = torch.optim.Adam(params, lr=args.lr)
    
    # 创建 CoTTA 适配器
    cotta = CoTTADepth(
        model=model,
        optimizer=optimizer,
        steps=args.cotta_steps,
        episodic=args.cotta_episodic,
        mt_alpha=args.mt_alpha,
        rst_m=args.rst_m,
        ap=args.ap,
        num_aug=args.num_aug,
        min_depth=args.min_depth,
        max_depth=args.max_depth
    )
    cotta.cuda()
    
    # 推理
    # 注意：SQLdepth 输出的是深度值（虽然键名是 "disp"），所以这里保存为 pred_depths
    pred_depths = []
    model.eval()  # 确保模型在评估模式
    
    # 检查是否在终端中运行
    is_tty = hasattr(sys.stderr, 'isatty') and sys.stderr.isatty()
    
    # 如果不在终端中（如通过 subprocess.PIPE），禁用进度条避免每行刷新
    # 或者使用简单的计数器
    if is_tty:
        # 在终端中：使用正常的进度条
        pbar = tqdm(
            dataloader, 
            desc="Evaluating", 
            ncols=100,
            file=sys.stderr,
            dynamic_ncols=False,
            leave=False
        )
        dataloader_iter = pbar
    else:
        # 不在终端中：使用简单的迭代器，每100个样本打印一次
        dataloader_iter = dataloader
        total = len(dataloader)
        print(f"Evaluating {total} samples...", file=sys.stderr, flush=True)
    
    for idx, batch in enumerate(dataloader_iter):
        # 非终端模式下，每100个样本打印一次进度
        if not is_tty and (idx + 1) % 100 == 0:
            print(f"Progress: {idx + 1}/{total} ({100*(idx+1)/total:.1f}%)", file=sys.stderr, flush=True)
        images = batch[("color", 0, 0)].cuda()
        
        if args.disable_cotta:
            # 直接使用原始模型预测（用于对比）
            with torch.no_grad():
                # 归一化输入
                if images.max() > 1.0:
                    images = images / 255.0
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(images.device)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(images.device)
                images_norm = (images - mean) / std
                
                pred_depth = model(images_norm)
                if isinstance(pred_depth, dict):
                    pred_depth = pred_depth[("disp", 0)]
        else:
            # CoTTA 适应和预测（返回深度值）
            pred_depth = cotta(images)
        
        pred_depth = pred_depth.detach().cpu().numpy()
        if pred_depth.ndim == 4:
            pred_depth = pred_depth[:, 0, :, :]
        elif pred_depth.ndim == 3:
            pred_depth = pred_depth
        pred_depths.append(pred_depth)
    
    pred_depths = np.concatenate(pred_depths, axis=0)
    
    # 评估
    gt_path = os.path.join(splits_dir, args.eval_split, "gt_depths.npz")
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]
    errors = []
    ratios = []
    MIN_DEPTH = args.min_depth
    MAX_DEPTH = args.max_depth
    
    for i in range(pred_depths.shape[0]):
        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]
        # SQLdepth 输出的是深度值，直接使用
        pred_depth = pred_depths[i]
        pred_depth = cv2.resize(pred_depth, (gt_width, gt_height))
        
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

