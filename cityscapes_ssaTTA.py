"""
SSA-TTA (Significant Subspace Alignment Test-Time Adaptation) for Depth Estimation on Cityscapes

本脚本基于 `cityscapes_completiontta.py`，将 Completion-TTA 替换为 SSA-TTA：
- 使用 tta_sqldepth_cityscapes_c.py 中的 RegressionSSA
- 数据集与评估流程参考 evaluate_cityscapes_vectta.py
- 模型结构与权重加载方式与 evaluate_cityscapes_vectta.py 保持一致
"""

import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm

from utils import readlines
from ssa_utils import (
    FeatureProjector,
    load_pca_stats,
    diagonal_gaussian_kl_loss,
    build_dim_weight,
)

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


class RegressionSSA(nn.Module):
    """基于 regression-tta 的显著子空间对齐模块（从 tta_sqldepth_cityscapes_c.py 复制）"""
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
        
        # 检查特征维度是否匹配
        # 特征维度应该等于 encoder 的 model_dim（经过池化后）
        # 使用一个小的 dummy input 来获取实际特征维度
        with torch.no_grad():
            device = next(depth_model.encoder.parameters()).device
            # 使用任意尺寸的 dummy input（池化后尺寸不影响特征维度）
            dummy_input = torch.zeros(1, 3, 64, 64).to(device)
            dummy_feature_map = depth_model.encoder(dummy_input)
            dummy_feature_vec = self.projector(dummy_feature_map)
            actual_feature_dim = dummy_feature_vec.shape[1]
        
        if mean.shape[0] != actual_feature_dim:
            raise ValueError(
                f"SSA 统计文件特征维度 ({mean.shape[0]}) 与当前模型特征维度 ({actual_feature_dim}) 不匹配！\n"
                f"请确保 SSA 统计文件是用相同的 model_dim 生成的。\n"
                f"当前模型 model_dim 为: {actual_feature_dim}\n"
                f"SSA 统计文件期望的 model_dim 为: {mean.shape[0]}\n"
                f"解决方案：使用 model_dim={mean.shape[0]} 重新生成 SSA 统计文件，或使用 model_dim={actual_feature_dim} 的统计文件。"
            )
        
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

    # SSA-TTA 参数（参考 tta_sqldepth_cityscapes_c.py）
    parser.add_argument('--ssa_stats_path', type=str, required=True,
                       help='SSA 特征统计文件路径（.pt 文件）')
    parser.add_argument('--ssa_topk', type=int, default=64, help='使用的 top-k 主成分数量')
    parser.add_argument('--ssa_steps', type=int, default=1, help='SSA-TTA 适应步数')
    parser.add_argument('--ssa_lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--ssa_update_mode', type=str,
                        choices=['encoder_bn', 'encoder_all', 'all'], default='encoder_bn',
                        help='更新模式：encoder_bn=仅Encoder的BN, encoder_all=整个Encoder, all=全部参数')
    parser.add_argument('--ssa_feature_pool', type=str, choices=['avg', 'max'], default='avg',
                       help='特征池化方式：avg=平均池化, max=最大池化')
    parser.add_argument('--ssa_eps', type=float, default=1e-6, help='数值稳定性参数')
    parser.add_argument('--ssa_weight_bias', type=float, default=1e-3, help='维度权重偏置')
    parser.add_argument('--ssa_weight_exp', type=float, default=1.0, help='维度权重指数')
    parser.add_argument('--ssa_grad_clip', type=float, default=0.0, help='梯度裁剪阈值（0表示不裁剪）')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader worker 数量')
    parser.add_argument('--disable_tta', action='store_true', help='禁用 SSA-TTA，直接使用原始模型预测（用于对比）')

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
    
    # 检查 SSA stats 文件
    if not args.disable_tta and not os.path.exists(args.ssa_stats_path):
        raise FileNotFoundError(f"SSA stats file not found: {args.ssa_stats_path}")

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

    # 创建 SSA-TTA 适配器
    if not args.disable_tta:
        ssa_tta = RegressionSSA(
            depth_model=depth_model,
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
        ).cuda()
        print("-> SSA-TTA enabled with update_mode={}, steps={}, lr={}, topk={}".format(
            args.ssa_update_mode, args.ssa_steps, args.ssa_lr, args.ssa_topk))
    else:
        ssa_tta = None
        depth_model.eval()
        print("-> SSA-TTA disabled, using original model")

    # 推理
    pred_depths = []

    print("-> Computing predictions with size {}x{}".format(HEIGHT, WIDTH))
    for batch in tqdm(dataloader, desc="Evaluating Cityscapes with SSA-TTA"):
        images = batch[("color", 0, 0)].cuda().float()

        if args.disable_tta:
            # 直接使用原始模型预测（用于对比）
            with torch.no_grad():
                if images.max() > 1.0:
                    images = images / 255.0
                mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
                std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
                images_norm = (images - mean) / std

                pred_depth = depth_model(images_norm)
                if isinstance(pred_depth, dict):
                    pred_depth = pred_depth[("disp", 0)]
        else:
            # SSA-TTA 适应和预测
            pred_depth = ssa_tta(images)

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
    print("\nSSA-TTA on SQLDepth (Cityscapes)")
    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")


if __name__ == "__main__":
    main()

