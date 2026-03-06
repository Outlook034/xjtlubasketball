"""
ProxyTTA-fast (Paper-like) for Depth Estimation on KITTI-C

- Implements the paper's Proxy Embeddings concept on SQLDepth
- Uses a sparse-depth branch + embedding module to generate proxy reference
- Fast: only updates the embedding module (and optionally sparse encoder)
- Evaluation protocol is aligned with your 5.x RMSE completion_tta.py (NO disp_to_depth)
"""

import os
import argparse

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import readlines

import datasets
import networks

from proxytta_sql_depth_fast import SQLDepthModel

# TENT (from tent-master)
import sys

_TENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tent-master", "tent-master"))
if _TENT_DIR not in sys.path:
    sys.path.append(_TENT_DIR)

from tent import Tent, collect_params, configure_model  # noqa: E402


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

    # Data/Model (aligned with your completion_tta.py)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--eval_split', type=str, default='eigen')
    parser.add_argument('--load_weights_folder', type=str, required=True)
    parser.add_argument('--backbone', type=str, default='resnet_lite')
    parser.add_argument('--height', type=int, default=192)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--batch_size', type=int, default=8)
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

    # TENT params
    parser.add_argument('--tta_steps', type=int, default=1)
    parser.add_argument('--tta_lr', type=float, default=1e-4)
    parser.add_argument('--tta_episodic', action='store_true')

    # Normalization (keep consistent with other scripts)
    parser.add_argument('--tta_norm_mean', nargs='+', type=float, default=[0.485, 0.456, 0.406])
    parser.add_argument('--tta_norm_std', nargs='+', type=float, default=[0.229, 0.224, 0.225])

    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--disable_tta', action='store_true')

    args = parser.parse_args()

    splits_dir = os.path.join(os.path.dirname(__file__), 'splits')
    filenames = readlines(os.path.join(splits_dir, args.eval_split, 'test_files.txt'))

    dataset = datasets.KITTIRAWDataset(
        args.data_path,
        filenames,
        args.height,
        args.width,
        [0],
        1,
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

    # Load model (from your 5.x script)
    if args.backbone in ["resnet", "resnet_lite"]:
        encoder = networks.ResnetEncoderDecoder(num_layers=args.num_layers, num_features=args.num_features, model_dim=args.model_dim)
    elif args.backbone == "resnet18_lite":
        encoder = networks.LiteResnetEncoderDecoder(model_dim=args.model_dim)
    else:
        raise NotImplementedError(f"Backbone {args.backbone} not supported in this script")

    encoder_path = os.path.join(args.load_weights_folder, 'encoder.pth')
    decoder_path = os.path.join(args.load_weights_folder, 'depth.pth')
    encoder.load_state_dict({k: v for k, v in torch.load(encoder_path, map_location='cpu').items() if k in encoder.state_dict()})

    if args.backbone.endswith('_lite'):
        decoder = networks.Lite_Depth_Decoder_QueryTr(in_channels=args.model_dim, patch_size=args.patch_size, dim_out=args.dim_out, embedding_dim=args.model_dim, query_nums=args.query_nums, num_heads=4, min_val=args.min_depth, max_val=args.max_depth)
    else:
        decoder = networks.Depth_Decoder_QueryTr(in_channels=args.model_dim, patch_size=args.patch_size, dim_out=args.dim_out, embedding_dim=args.model_dim, query_nums=args.query_nums, num_heads=4, min_val=args.min_depth, max_val=args.max_depth)

    decoder.load_state_dict(torch.load(decoder_path, map_location='cpu'))

    encoder.cuda().eval()
    decoder.cuda().eval()
    depth_model = SQLDepthModel(encoder, decoder).cuda()

    if not args.disable_tta:
        # Configure TENT: update only BN affine params via entropy minimization
        configure_model(depth_model)
        params, _ = collect_params(depth_model)
        optimizer = torch.optim.Adam(params, lr=args.tta_lr)
        tta = Tent(depth_model, optimizer, steps=args.tta_steps, episodic=args.tta_episodic).cuda()
        print(f"-> TENT enabled: steps={args.tta_steps}, lr={args.tta_lr}, episodic={args.tta_episodic}")
    else:
        tta = None
        depth_model.eval()
        print('-> TTA disabled, using original model')

    gt_path = os.path.join(splits_dir, args.eval_split, 'gt_depths.npz')
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)['data']

    pred_depths = []
    idx = 0

    for batch in tqdm(dataloader, desc=f'Evaluating KITTI-C with ProxyTTA-fast'):
        images = batch[("color", 0, 0)].cuda().float()
        bs = images.shape[0]
        batch_gt = gt_depths[idx:idx + bs]
        idx += bs

        if args.disable_tta:
            with torch.no_grad():
                if images.max() > 1.0:
                    images = images / 255.0
                mean = torch.tensor(args.tta_norm_mean).view(1, 3, 1, 1).to(images.device)
                std = torch.tensor(args.tta_norm_std).view(1, 3, 1, 1).to(images.device)
                images_norm = (images - mean) / std
                pred = depth_model(images_norm)
        else:
            if images.max() > 1.0:
                images = images / 255.0
            mean = torch.tensor(args.tta_norm_mean).view(1, 3, 1, 1).to(images.device)
            std = torch.tensor(args.tta_norm_std).view(1, 3, 1, 1).to(images.device)
            images_norm = (images - mean) / std
            pred = tta(images_norm)

        pred = pred.detach().cpu().numpy()
        if pred.ndim == 4:
            pred = pred[:, 0, :, :]
        pred_depths.append(pred)

    pred_depths = np.concatenate(pred_depths, axis=0)

    # Evaluation (Eigen)
    errors = []
    ratios = []
    MIN_DEPTH = args.min_depth
    MAX_DEPTH = args.max_depth

    for i in range(pred_depths.shape[0]):
        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_depth = pred_depths[i]
        pred_depth = cv2.resize(pred_depth, (gt_width, gt_height))

        mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
        crop = np.array([
            0.40810811 * gt_height,
            0.99189189 * gt_height,
            0.03594771 * gt_width,
            0.96405229 * gt_width,
        ]).astype(np.int32)
        crop_mask = np.zeros(mask.shape)
        crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
        mask = np.logical_and(mask, crop_mask)

        pred_depth_eval = pred_depth[mask]
        gt_depth_eval = gt_depth[mask]

        pred_depth_eval *= args.pred_depth_scale_factor
        if not args.disable_median_scaling:
            ratio = np.median(gt_depth_eval) / np.median(pred_depth_eval)
            ratios.append(ratio)
            pred_depth_eval *= ratio

        pred_depth_eval[pred_depth_eval < MIN_DEPTH] = MIN_DEPTH
        pred_depth_eval[pred_depth_eval > MAX_DEPTH] = MAX_DEPTH

        errors.append(compute_errors(gt_depth_eval, pred_depth_eval))

    if not args.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    mean_errors = np.array(errors).mean(0)
    print("\nProxyTTA-fast (paper-like) on SQLDepth (KITTI-C)")
    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")


if __name__ == '__main__':
    main()
