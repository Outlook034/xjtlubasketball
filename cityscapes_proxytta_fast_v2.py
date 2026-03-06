"""
ProxyTTA-fast v2 for Depth Estimation on Cityscapes

- Independent entry script (does NOT modify existing scripts)
- Mirrors cityscapes_completiontta.py data/metrics pipeline
- Uses GT depth ONLY to *sample sparse depth* for adaptation (same as your current Completion-TTA script)
- Updates ONLY an adaptation layer (fast)

This script is intended for fair comparison against your current:
- baseline (--disable_tta)
- Completion-TTA (cityscapes_completiontta.py)

Output is treated consistently with your SQLDepth scripts ("disp",0).
"""

import os
import sys
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm

from utils import readlines

# 使得可以导入上级目录的 datasets / networks
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import datasets
import networks

from completion_tta import create_sparse_depth
from proxytta_fast_v2 import SQLDepthFeatureModel, ProxyTTAFastV2Config, ProxyTTAFastV2Module
from layers import disp_to_depth


def compute_errors(gt, pred):
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
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield str(arg)


def main():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.convert_arg_line_to_args = convert_arg_line_to_args

    # 数据与模型
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--eval_data_path', type=str, default=None)
    parser.add_argument('--load_weights_folder', type=str, default=None)
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

    # TTA params
    parser.add_argument('--tta_steps', type=int, default=1)
    parser.add_argument('--tta_lr', type=float, default=4e-5)
    parser.add_argument('--tta_momentum_teacher', type=float, default=0.999)
    parser.add_argument('--tta_w_sparse', type=float, default=1.0)
    parser.add_argument('--tta_w_smooth', type=float, default=1.0)
    parser.add_argument('--tta_w_cos', type=float, default=0.0)
    parser.add_argument('--tta_grad_clip', type=float, default=5.0)
    parser.add_argument('--sparse_keep_ratio', type=float, default=0.1)
    parser.add_argument('--tta_norm_mean', nargs='+', type=float, default=[0.485, 0.456, 0.406])
    parser.add_argument('--tta_norm_std', nargs='+', type=float, default=[0.229, 0.224, 0.225])

    # Augmentation (same arg names as completion script)
    parser.add_argument('--tta_aug_prob', type=float, default=0.5)
    parser.add_argument('--tta_aug_flip_type', nargs='+', type=str, default=['none'])
    parser.add_argument('--tta_aug_rotate', type=float, default=0.0)
    parser.add_argument('--tta_aug_brightness', nargs='+', type=float, default=[-1, -1])
    parser.add_argument('--tta_aug_contrast', nargs='+', type=float, default=[-1, -1])
    parser.add_argument('--tta_aug_gamma', nargs='+', type=float, default=[-1, -1])
    parser.add_argument('--tta_aug_saturation', nargs='+', type=float, default=[-1, -1])
    parser.add_argument('--tta_aug_noise_type', type=str, default='none')
    parser.add_argument('--tta_aug_noise_spread', type=float, default=-1)

    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--disable_tta', action='store_true')

    args = parser.parse_args()

    if args.eval_data_path is not None:
        args.data_path = args.eval_data_path
    if args.data_path is None:
        raise ValueError('必须提供 --data_path 或 --eval_data_path')
    if args.load_weights_folder is None:
        raise ValueError('必须提供 --load_weights_folder')

    args.load_weights_folder = os.path.expanduser(args.load_weights_folder)
    if not os.path.isabs(args.load_weights_folder):
        args.load_weights_folder = os.path.abspath(args.load_weights_folder)

    splits_dir = os.path.join(os.path.dirname(__file__), 'splits')
    test_files_path = os.path.join(splits_dir, args.eval_split, 'test_files.txt')
    if not os.path.exists(test_files_path):
        raise FileNotFoundError(f'Test files not found: {test_files_path}')
    filenames = readlines(test_files_path)

    encoder_path = os.path.join(args.load_weights_folder, 'encoder.pth')
    decoder_path = os.path.join(args.load_weights_folder, 'depth.pth')
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

    # Build encoder/decoder
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

    depth_model = SQLDepthFeatureModel(encoder, depth_decoder).cuda()

    if not args.disable_tta:
        cfg = ProxyTTAFastV2Config(
            steps=args.tta_steps,
            lr=args.tta_lr,
            momentum_teacher=args.tta_momentum_teacher,
            w_sparse=args.tta_w_sparse,
            w_smooth=args.tta_w_smooth,
            w_cos=args.tta_w_cos,
            grad_clip=args.tta_grad_clip,
            norm_mean=tuple(args.tta_norm_mean),
            norm_std=tuple(args.tta_norm_std),
            aug_prob=args.tta_aug_prob,
            aug_flip_type=args.tta_aug_flip_type,
            aug_rotate=args.tta_aug_rotate,
            aug_brightness=args.tta_aug_brightness,
            aug_contrast=args.tta_aug_contrast,
            aug_gamma=args.tta_aug_gamma,
            aug_saturation=args.tta_aug_saturation,
            aug_noise_type=args.tta_aug_noise_type,
            aug_noise_spread=args.tta_aug_noise_spread,
        )
        tta = ProxyTTAFastV2Module(depth_model, cfg).cuda()
        print(f'-> ProxyTTA-fast v2 enabled: steps={cfg.steps}, lr={cfg.lr}, w_sparse={cfg.w_sparse}, w_smooth={cfg.w_smooth}, w_cos={cfg.w_cos}')
    else:
        tta = None
        depth_model.eval()
        print('-> TTA disabled, using original model')

    gt_depths_dir = os.path.join(splits_dir, args.eval_split, 'gt_depths')
    if not os.path.isdir(gt_depths_dir):
        raise FileNotFoundError(f'Cityscapes gt_depths directory not found: {gt_depths_dir}')

    pred_depths = []
    idx0 = 0

    print(f'-> Computing predictions with size {HEIGHT}x{WIDTH}')
    for batch in tqdm(dataloader, desc='Evaluating Cityscapes with ProxyTTA-fast v2'):
        images = batch[("color", 0, 0)].cuda().float()
        bs = images.shape[0]

        # Load GT for sparse sampling (same as completion script)
        batch_gt_list = []
        for b in range(bs):
            gt_path = os.path.join(gt_depths_dir, f"{idx0 + b:03d}_depth.npy")
            if not os.path.exists(gt_path):
                raise FileNotFoundError(f'GT depth not found: {gt_path}')
            gt_depth = np.load(gt_path)
            gt_height, _ = gt_depth.shape[:2]
            gt_height = int(round(gt_height * 0.75))
            gt_depth = gt_depth[:gt_height]
            batch_gt_list.append(gt_depth)
        batch_gt = np.stack(batch_gt_list, axis=0)
        idx0 += bs

        if args.disable_tta:
            with torch.no_grad():
                # normalize here
                if images.max() > 1.0:
                    images = images / 255.0
                mean = torch.tensor(args.tta_norm_mean).view(1, 3, 1, 1).to(images.device)
                std = torch.tensor(args.tta_norm_std).view(1, 3, 1, 1).to(images.device)
                disp, _ = depth_model((images - mean) / std)
                pred, _ = disp_to_depth(disp, args.min_depth, args.max_depth)
        else:
            # Resize GT to input size and sample sparse depth
            batch_gt_resized = []
            for gt in batch_gt:
                gt_resized = cv2.resize(gt, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)
                batch_gt_resized.append(gt_resized)
            batch_gt_resized = np.stack(batch_gt_resized, axis=0)

            sparse_depth, _valid = create_sparse_depth(
                batch_gt_resized,
                keep_ratio=args.sparse_keep_ratio,
                device=images.device
            )

            disp = tta.adapt(images, sparse_depth)
            pred, _ = disp_to_depth(disp, args.min_depth, args.max_depth)

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
        gt_depth = np.load(gt_path)
        gt_height, gt_width = gt_depth.shape[:2]

        gt_height = int(round(gt_height * 0.75))
        gt_depth = gt_depth[:gt_height]

        pred_depth = pred_depths[i]
        pred_depth = cv2.resize(pred_depth, (gt_width, gt_height))

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
    print('\nProxyTTA-fast v2 on SQLDepth (Cityscapes)')
    print('\n  ' + ('{:>8} | ' * 7).format('abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3'))
    print(('&{: 8.3f}  ' * 7).format(*mean_errors.tolist()) + '\\\\')
    print('\n-> Done!')


if __name__ == '__main__':
    main()
