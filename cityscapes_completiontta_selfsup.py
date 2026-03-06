"""
Completion-TTA (Self-Supervised) for Depth Estimation on Cityscapes

- Independent entry script (does NOT modify existing scripts)
- Mirrors cityscapes_completiontta.py but removes GT-based sparse depth supervision
- Uses only smoothness loss (and optionally teacher consistency) for TTA
- Designed for fair comparison with VecTTA and ProxyTTA-fast (no GT in adaptation)

Note: This version does NOT use any GT during adaptation; GT is only used for evaluation.
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
from pathlib import Path

from utils import readlines

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import datasets
import networks

from completion_tta import CompletionTTAModule, SQLDepthModel


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
    parser.convert_arg_line_to_args = convert_arg_line_to_args

    # 数据与模型
    parser.add_argument('--data_path', type=str, default=None, help='Cityscapes 数据根目录')
    parser.add_argument('--eval_data_path', type=str, default=None, help='Cityscapes 数据根目录（兼容参数文件）')
    parser.add_argument('--load_weights_folder', type=str, default=None, help='模型权重文件夹路径')
    parser.add_argument('--eval_split', type=str, default='cityscapes')
    parser.add_argument('--height', type=int, default=192)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--model_dim', type=int, default=32)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--dim_out', type=int, default=128)
    parser.add_argument('--query_nums', type=int, default=128)
    parser.add_argument('--min_depth', type=float, default=1e-3)
    parser.add_argument('--max_depth', float, default=80.0)
    parser.add_argument('--disable_median_scaling', action='store_true')
    parser.add_argument('--pred_depth_scale_factor', type=float, default=1.0)

    # Completion-TTA 参数（自监督版：去掉 sparse depth 相关）
    parser.add_argument('--tta_steps', type=int, default=1, help='Completion-TTA 适应步数')
    parser.add_argument('--tta_lr', type=float, default=4e-5, help='学习率')
    parser.add_argument('--tta_update_mode', type=str, choices=['bn', 'bn_decoder', 'all'], default='bn_decoder',
                        help='更新模式：bn=仅BatchNorm, bn_decoder=BatchNorm+Decoder, all=全部参数')
    parser.add_argument('--tta_w_smooth', type=float, default=1.0, help='平滑损失权重')
    parser.add_argument('--tta_grad_clip', type=float, default=5.0, help='梯度裁剪阈值')
    parser.add_argument('--tta_norm_mean', nargs='+', type=float, default=[0.485, 0.456, 0.406],
                        help='图像归一化均值')
    parser.add_argument('--tta_norm_std', nargs='+', type=float, default=[0.229, 0.224, 0.225],
                        help='图像归一化标准差')
    parser.add_argument('--tta_aug_prob', type=float, default=0.5, help='数据增强概率')
    parser.add_argument('--tta_aug_flip_type', nargs='+', type=str, default=['none'], help='翻转类型')
    parser.add_argument('--tta_aug_rotate', type=float, default=0.0, help='最大旋转角度')
    parser.add_argument('--tta_aug_brightness', nargs='+', type=float, default=[-1, -1], help='亮度增强范围')
    parser.add_argument('--tta_aug_contrast', nargs='+', type=float, default=[-1, -1], help='对比度增强范围')
    parser.add_argument('--tta_aug_gamma', nargs='+', type=float, default=[-1, -1], help='Gamma 增强范围')
    parser.add_argument('--tta_aug_saturation', nargs='+', type=float, default=[-1, -1], help='饱和度增强范围')
    parser.add_argument('--tta_aug_noise_type', type=str, default='none', help='噪声类型')
    parser.add_argument('--tta_aug_noise_spread', type=float, default=-1, help='噪声扩散')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader worker 数量')
    parser.add_argument('--disable_tta', action='store_true', help='禁用 Completion-TTA，直接使用原始模型预测（用于对比）')

    args = parser.parse_args()

    # 统一处理 data_path 和 eval_data_path（兼容参数文件）
    if args.eval_data_path is not None:
        args.data_path = args.eval_data_path
    if args.data_path is None:
        raise ValueError('必须提供 --data_path 或 --eval_data_path 参数')

    if args.load_weights_folder is None:
        raise ValueError('必须提供 --load_weights_folder 参数')

    args.load_weights_folder = os.path.expanduser(args.load_weights_folder)
    if not os.path.isabs(args.load_weights_folder):
        args.load_weights_folder = os.path.abspath(args.load_weights_folder)

    # 数据集与文件列表
    splits_dir = os.path.join(os.path.dirname(__file__), 'splits')
    test_files_path = os.path.join(splits_dir, args.eval_split, 'test_files.txt')
    if not os.path.exists(test_files_path):
        raise FileNotFoundError(f'Test files not found: {test_files_path}')

    filenames = readlines(test_files_path)

    # Load weights
    encoder_path = os.path.join(args.load_weights_folder, 'encoder.pth')
    decoder_path = os.path.join(args.load_weights_folder, 'depth.pth')
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f'Encoder weights not found: {encoder_path}')
    if not os.path.exists(decoder_path):
        raise FileNotFoundError(f'Decoder weights not found: {decoder_path}')

    encoder_dict = torch.load(encoder_path)
    try:
        HEIGHT, WIDTH = encoder_dict['height'], encoder_dict['width']
    except KeyError:
        HEIGHT, WIDTH = args.height, args.width

    dataset = datasets.CityscapesEvalDataset(
        args.data_path,
        filenames,
        HEIGHT,
        WIDTH,
        [0],
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

    # Infer model_dim from encoder checkpoint
    encoder_state = encoder_dict if isinstance(encoder_dict, dict) else {}
    inferred_model_dim = None
    for k, v in encoder_state.items():
        if k.endswith('decoder.conv3.weight') and hasattr(v, 'shape') and len(v.shape) == 4:
            inferred_model_dim = int(v.shape[0])
            break
    if inferred_model_dim is None:
        inferred_model_dim = args.model_dim

    # Build model
    encoder = networks.Resnet50EncoderDecoder(model_dim=inferred_model_dim)

    decoder_dict = torch.load(decoder_path)
    if 'convert_to_prob.0.weight' in decoder_dict:
        actual_dim_out = decoder_dict['convert_to_prob.0.weight'].shape[0]
        actual_query_nums = decoder_dict['convert_to_prob.0.weight'].shape[1]
    else:
        actual_dim_out = args.dim_out
        actual_query_nums = args.query_nums

    depth_decoder = networks.Depth_Decoder_QueryTr(
        in_channels=inferred_model_dim,
        patch_size=args.patch_size,
        dim_out=actual_dim_out,
        embedding_dim=inferred_model_dim,
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

    # Create Completion-TTA adapter (self-supervised version)
    if not args.disable_tta:
        # Override sparse depth weight to zero; we will not use sparse depth in adaptation
        args.tta_w_sparse = 0.0
        # We do NOT provide sparse depth during adapt; CompletionTTAModule will skip that loss
        completion_tta = CompletionTTAModule(depth_model, args).cuda()
        print(f'-> Completion-TTA (self-supervised) enabled with update_mode={args.tta_update_mode}, steps={args.tta_steps}, lr={args.tta_lr}')
        print('    Note: sparse depth supervision is disabled (no GT used in adaptation)')
    else:
        completion_tta = None
        depth_model.eval()
        print('-> Completion-TTA disabled, using original model')

    # Load GT depths for evaluation only (not used in adaptation)
    gt_depths_dir = os.path.join(splits_dir, args.eval_split, 'gt_depths')
    if not os.path.isdir(gt_depths_dir):
        raise FileNotFoundError(f'Cityscapes gt_depths directory not found: {gt_depths_dir}')

    # Inference
    pred_depths = []

    print(f'-> Computing predictions with size {HEIGHT}x{WIDTH}')
    for batch in tqdm(dataloader, desc='Evaluating Cityscapes with Completion-TTA (self-supervised)'):
        images = batch[("color", 0, 0)].cuda().float()

        if args.disable_tta:
            # Direct inference
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
            # Self-supervised adaptation: we pass None for sparse_depth
            pred_depth = completion_tta.adapt(images, sparse_depth=None)

        pred_depth = pred_depth.detach().cpu().numpy()
        if pred_depth.ndim == 4:
            pred_depth = pred_depth[:, 0, :, :]
        pred_depths.append(pred_depth)

    pred_depths = np.concatenate(pred_depths, axis=0)

    # Evaluation (same as cityscapes_completiontta.py)
    errors = []
    ratios = []
    MIN_DEPTH = args.min_depth
    MAX_DEPTH = args.max_depth

    print('-> Evaluating on Cityscapes')
    for i in tqdm(range(pred_depths.shape[0]), desc='Computing metrics'):
        gt_path = os.path.join(gt_depths_dir, f"{i:03d}_depth.npy")
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f'GT depth not found: {gt_path}')
        gt_depth = np.load(gt_path)
        gt_height, gt_width = gt_depth.shape[:2]

        # Remove bottom 25%
        gt_height = int(round(gt_height * 0.75))
        gt_depth = gt_depth[:gt_height]

        pred_depth = pred_depths[i]
        pred_depth = cv2.resize(pred_depth, (gt_width, gt_height))

        # Centre crop to middle 50%
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
        print(' Scaling ratios | med: {:0.3f} | std: {:0.3f}'.format(med, np.std(ratios / med)))

    mean_errors = np.array(errors).mean(0)
    print('\nCompletion-TTA (self-supervised) on SQLDepth (Cityscapes)')
    print('\n  ' + ('{:>8} | ' * 7).format('abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3'))
    print(('&{: 8.3f}  ' * 7).format(*mean_errors.tolist()) + '\\\\')
    print('\n-> Done!')


if __name__ == '__main__':
    main()