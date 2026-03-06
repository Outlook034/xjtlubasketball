"""
CoTTA (Continual Test-Time Adaptation) for Depth Estimation on Cityscapes

本脚本是基于 `cotta_sqldepth.py` 的 Cityscapes 版本：
- 数据集与评估流程参考 `evaluate_cityscapes_vectta.py`
- 模型结构与权重加载方式与 `evaluate_cityscapes_vectta.py` 保持一致
"""

import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from copy import deepcopy
from PIL import Image
import argparse
from tqdm import tqdm

from utils import readlines
from layers import disp_to_depth

# 使得可以导入上级目录的 datasets / networks
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import datasets
import networks


class DepthAugmentation:
    """针对深度估计的数据增强（与 cotta_sqldepth.py 一致）"""

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
        img = F.pad(img.unsqueeze(0), (padding, padding, padding, padding), mode="replicate")
        img = img.squeeze(0)

        # 随机仿射变换
        if self.soft:
            angle = np.random.uniform(-8, 8)
            translate = (np.random.uniform(-w / 16, w / 16), np.random.uniform(-h / 16, h / 16))
            scale = np.random.uniform(0.95, 1.05)
        else:
            angle = np.random.uniform(-15, 15)
            translate = (np.random.uniform(-w / 16, w / 16), np.random.uniform(-h / 16, h / 16))
            scale = np.random.uniform(0.9, 1.1)

        img = transforms.functional.affine(
            img,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=0,
            interpolation=transforms.InterpolationMode.BILINEAR,
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
    """SQLdepth 模型包装器（与 cotta_sqldepth.py 一致）"""

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
    CoTTA for Depth Estimation（与 cotta_sqldepth.py 基本一致）
    """

    def __init__(
        self,
        model,
        optimizer,
        steps=1,
        episodic=False,
        mt_alpha=0.999,
        rst_m=0.001,
        ap=0.9,
        num_aug=32,
        min_depth=0.1,
        max_depth=80.0,
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "cotta requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        # 保存模型和优化器状态
        (
            self.model_state,
            self.optimizer_state,
            self.model_ema,
            self.model_anchor,
        ) = self.copy_model_and_optimizer(self.model, self.optimizer)

        # 数据增强
        self.transform = get_depth_tta_transforms(soft=False)

        # 超参数
        self.mt = mt_alpha  # EMA 更新率
        self.rst = rst_m  # 随机恢复概率
        self.ap = ap  # 置信度阈值
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
        """前向传播，返回深度（SQLdepth 输出的是深度值，虽然键名是 \"disp\"）"""
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
        self.load_model_and_optimizer(
            self.model,
            self.optimizer,
            self.model_state,
            self.optimizer_state,
        )
        # 同时重置 teacher 模型
        (
            self.model_state,
            self.optimizer_state,
            self.model_ema,
            self.model_anchor,
        ) = self.copy_model_and_optimizer(self.model, self.optimizer)

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
            alpha_teacher=self.mt,
        )

        # 随机恢复（stochastic restore）
        for nm, m in self.model.named_modules():
            for npp, p in m.named_parameters():
                if npp in ["weight", "bias"] and p.requires_grad:
                    mask = (torch.rand(p.shape, device=p.device) < self.rst).float()
                    with torch.no_grad():
                        state_key = f"{nm}.{npp}"
                        if state_key in self.model_state:
                            p.data = self.model_state[state_key] * mask + p * (1.0 - mask)

        # 不返回任何值，因为 forward 方法会单独获取最终预测
        return None


def collect_params(model):
    """收集所有可训练参数"""
    params = []
    names = []
    for nm, m in model.named_modules():
        for np_name, p in m.named_parameters():
            if np_name in ["weight", "bias"] and p.requires_grad:
                params.append(p)
                names.append(f"{nm}.{np_name}")
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

    # CoTTA 参数
    parser.add_argument('--cotta_steps', type=int, default=1, help='CoTTA 适应步数')
    parser.add_argument('--cotta_episodic', action='store_true', help='是否使用 episodic 模式')
    parser.add_argument('--mt_alpha', type=float, default=0.999, help='EMA 更新率')
    parser.add_argument('--rst_m', type=float, default=0.001, help='随机恢复概率')
    parser.add_argument('--ap', type=float, default=0.9, help='置信度阈值')
    parser.add_argument('--num_aug', type=int, default=32, help='数据增强次数')
    parser.add_argument('--lr', type=float, default=1e-5, help='学习率（建议较小值，如1e-5）')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader worker 数量')
    parser.add_argument('--disable_cotta', action='store_true', help='禁用 CoTTA，直接使用原始模型预测（用于对比）')

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

    # 配置模型用于 CoTTA
    depth_model = configure_model(depth_model)

    # 收集参数并创建优化器
    params, _ = collect_params(depth_model)
    optimizer = torch.optim.Adam(params, lr=args.lr)

    # 创建 CoTTA 适配器
    cotta = CoTTADepth(
        model=depth_model,
        optimizer=optimizer,
        steps=args.cotta_steps,
        episodic=args.cotta_episodic,
        mt_alpha=args.mt_alpha,
        rst_m=args.rst_m,
        ap=args.ap,
        num_aug=args.num_aug,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
    )
    cotta.cuda()

    # 推理
    pred_depths = []
    depth_model.eval()  # 评估模式

    for batch in tqdm(dataloader, desc="Evaluating Cityscapes with CoTTA"):
        images = batch[("color", 0, 0)].cuda()

        if args.disable_cotta:
            # 直接使用原始模型预测（用于对比）
            with torch.no_grad():
                if images.max() > 1.0:
                    images = images / 255.0
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(images.device)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(images.device)
                images_norm = (images - mean) / std

                pred_depth = depth_model(images_norm)
                if isinstance(pred_depth, dict):
                    pred_depth = pred_depth[("disp", 0)]
        else:
            # CoTTA 适应和预测（返回深度值）
            pred_depth = cotta(images)

        pred_depth = pred_depth.detach().cpu().numpy()
        if pred_depth.ndim == 4:
            pred_depth = pred_depth[:, 0, :, :]
        pred_depths.append(pred_depth)

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

        pred_depth = pred_depths[i]
        pred_depth = cv2.resize(pred_depth, (gt_width, gt_height))

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
    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")


if __name__ == "__main__":
    main()


