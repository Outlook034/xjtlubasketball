"""
ProxyTTA-fast (EMA teacher) for Depth Estimation on Cityscapes

- Independent entry script (does NOT modify existing scripts)
- Uses ProxyTTAFastModule from proxytta_fast.py
- Evaluation protocol follows cityscapes_completiontta.py / evaluate_cityscapes_vectta.py

Note: We keep the output consistent with existing SQLDepth usage (treating the model
output ("disp",0) as the value being evaluated), per user request.
"""

import os
import sys
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
from pathlib import Path

from utils import readlines

# 使得可以导入上级目录的 datasets / networks
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import datasets
import networks

from proxytta_fast import (
    SQLDepthFeatureModel,
    ProxyTTAFastConfig,
    ProxyTTAFastModule,
)


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
    parser.add_argument('--max_depth', type=float, default=80.0)
    parser.add_argument('--disable_median_scaling', action='store_true')
    parser.add_argument('--pred_depth_scale_factor', type=float, default=1.0)

    # ProxyTTA-fast params
    parser.add_argument('--tta_steps', type=int, default=1)
    parser.add_argument('--tta_lr', type=float, default=1e-4)
    parser.add_argument('--tta_momentum_teacher', type=float, default=0.999)
    parser.add_argument('--tta_w_cos', type=float, default=1.0)
    parser.add_argument('--tta_w_smooth', type=float, default=1.0)
    parser.add_argument('--tta_grad_clip', type=float, default=5.0)
    parser.add_argument('--tta_norm_mean', nargs='+', type=float, default=[0.485, 0.456, 0.406])
    parser.add_argument('--tta_norm_std', nargs='+', type=float, default=[0.229, 0.224, 0.225])
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--disable_tta', action='store_true', help='禁用 ProxyTTA-fast，直接使用原始模型预测（对比）')

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

    # Infer model_dim from encoder checkpoint to avoid decoder shape mismatches
    encoder_state = encoder_dict if isinstance(encoder_dict, dict) else {}
    inferred_model_dim = None
    for k, v in encoder_state.items():
        if k.endswith('decoder.conv3.weight') and hasattr(v, 'shape') and len(v.shape) == 4:
            inferred_model_dim = int(v.shape[0])
            break
    if inferred_model_dim is None:
        inferred_model_dim = args.model_dim

    # Build model (same pattern as cityscapes_completiontta.py)
    encoder = networks.Resnet50EncoderDecoder(model_dim=inferred_model_dim)

    decoder_dict = torch.load(decoder_path)
    if 'convert_to_prob.0.weight' in decoder_dict:
        actual_dim_out = decoder_dict['convert_to_prob.0.weight'].shape[0]
        actual_query_nums = decoder_dict['convert_to_prob.0.weight'].shape[1]
    else:
        actual_dim_out = args.dim_out
        actual_query_nums = args.query_nums

    # Depth decoder must match encoder feature channels and embedding_dim used in checkpoint
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

    depth_model = SQLDepthFeatureModel(encoder, depth_decoder).cuda()

    # ProxyTTA-fast module
    if not args.disable_tta:
        cfg = ProxyTTAFastConfig(
            steps=args.tta_steps,
            lr=args.tta_lr,
            momentum_teacher=args.tta_momentum_teacher,
            w_cos=args.tta_w_cos,
            w_smooth=args.tta_w_smooth,
            grad_clip=args.tta_grad_clip,
            norm_mean=tuple(args.tta_norm_mean),
            norm_std=tuple(args.tta_norm_std),
        )
        tta = ProxyTTAFastModule(depth_model, cfg).cuda()
        print(f'-> ProxyTTA-fast enabled: steps={cfg.steps}, lr={cfg.lr}, w_cos={cfg.w_cos}, w_smooth={cfg.w_smooth}')
    else:
        tta = None
        depth_model.eval()
        print('-> ProxyTTA-fast disabled, using original model')

    # Load GT depths for evaluation
    gt_depths_dir = os.path.join(splits_dir, args.eval_split, 'gt_depths')
    if not os.path.isdir(gt_depths_dir):
        raise FileNotFoundError(f'Cityscapes gt_depths directory not found: {gt_depths_dir}')

    # Inference
    pred_depths = []

    print(f'-> Computing predictions with size {HEIGHT}x{WIDTH}')
    for idx, batch in enumerate(tqdm(dataloader, desc='Evaluating Cityscapes with ProxyTTA-fast')):
        images = batch[("color", 0, 0)].cuda().float()

        if args.disable_tta:
            with torch.no_grad():
                # Keep consistent with existing scripts: model output is treated as pred_depth
                disp, _ = depth_model(_normalize_images(images, args.tta_norm_mean, args.tta_norm_std))
                pred = disp
        else:
            pred = tta.adapt(images)

        pred = pred.detach().cpu().numpy()
        if pred.ndim == 4:
            pred = pred[:, 0, :, :]
        pred_depths.append(pred)

    pred_depths = np.concatenate(pred_depths, axis=0)

    # Evaluation
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
    print('\nProxyTTA-fast on SQLDepth (Cityscapes)')
    print('\n  ' + ('{:>8} | ' * 7).format('abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3'))
    print(('&{: 8.3f}  ' * 7).format(*mean_errors.tolist()) + '\\\\')
    print('\n-> Done!')


def _normalize_images(images: torch.Tensor, mean_list, std_list) -> torch.Tensor:
    if images.max() > 1.0:
        images = images / 255.0
    mean = torch.tensor(mean_list, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(std_list, device=images.device).view(1, 3, 1, 1)
    return (images - mean) / std


if __name__ == '__main__':
    main()
