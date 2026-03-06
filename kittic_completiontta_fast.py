"""
Completion-TTA Fast (Adaptation Layer Only) for Depth Estimation on KITTI-C

- Independent entry script (does NOT modify existing scripts)
- Mirrors completion_tta.py's KITTI(Eigen) evaluation logic
- Uses CompletionTTAFastModule (only updates an adaptation layer)
- Intended to run on KITTI-C corruption folders (data_path points to .../kitti_data)

Evaluation protocol:
- Uses splits/eigen/test_files.txt
- Loads splits/eigen/gt_depths.npz
- Applies Eigen crop mask
- Uses disp_to_depth before evaluation (official baseline convention)

GT depth is used only to sample sparse depth (same as completion_tta.py).
"""

import os
import argparse

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import readlines

import datasets
import networks

from completion_tta import create_sparse_depth
from completion_tta_fast_kittic import CompletionTTAFastModule


class SQLDepthModel(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        out = self.decoder(self.encoder(x))
        if isinstance(out, dict):
            return out[("disp", 0)]
        return out


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

    parser.add_argument('--data_path', type=str, required=True, help='KITTI-C corruption root that contains kitti_raw_data structure')
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

    # Fast completion TTA params
    parser.add_argument('--tta_steps', type=int, default=1)
    parser.add_argument('--tta_lr', type=float, default=4e-5)
    parser.add_argument('--tta_w_sparse', type=float, default=1.0)
    parser.add_argument('--tta_w_smooth', type=float, default=1.0)
    parser.add_argument('--tta_grad_clip', type=float, default=5.0)
    parser.add_argument('--sparse_keep_ratio', type=float, default=0.1)
    parser.add_argument('--tta_norm_mean', nargs='+', type=float, default=[0.485, 0.456, 0.406])
    parser.add_argument('--tta_norm_std', nargs='+', type=float, default=[0.229, 0.224, 0.225])
    parser.add_argument('--tta_aug_prob', type=float, default=0.0)
    parser.add_argument('--tta_aug_crop_type', nargs='+', type=str, default=['none'])
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

    # Load model
    encoder_path = os.path.join(args.load_weights_folder, 'encoder.pth')
    decoder_path = os.path.join(args.load_weights_folder, 'depth.pth')

    if args.backbone in ["resnet", "resnet_lite"]:
        encoder = networks.ResnetEncoderDecoder(
            num_layers=args.num_layers,
            num_features=args.num_features,
            model_dim=args.model_dim,
        )
    elif args.backbone == "resnet18_lite":
        encoder = networks.LiteResnetEncoderDecoder(model_dim=args.model_dim)
    elif args.backbone == "eff_b5":
        encoder = networks.BaseEncoder.build(num_features=args.num_features, model_dim=args.model_dim)
    else:
        encoder = networks.Unet(
            pretrained=False,
            backbone=args.backbone,
            in_channels=3,
            num_classes=args.model_dim,
            decoder_channels=[1024, 512, 256, 128],
        )

    encoder_dict = torch.load(encoder_path, map_location='cpu')
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in encoder.state_dict()})

    if args.backbone.endswith('_lite'):
        decoder = networks.Lite_Depth_Decoder_QueryTr(
            in_channels=args.model_dim,
            patch_size=args.patch_size,
            dim_out=args.dim_out,
            embedding_dim=args.model_dim,
            query_nums=args.query_nums,
            num_heads=4,
            min_val=args.min_depth,
            max_val=args.max_depth,
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
            max_val=args.max_depth,
        )

    decoder.load_state_dict(torch.load(decoder_path, map_location='cpu'))

    encoder.cuda().eval()
    decoder.cuda().eval()

    depth_model = SQLDepthModel(encoder, decoder).cuda()

    if not args.disable_tta:
        tta = CompletionTTAFastModule(depth_model, args).cuda()
        print(f"-> Completion-TTA Fast (KITTI-C) enabled: steps={args.tta_steps}, lr={args.tta_lr}")
    else:
        tta = None
        depth_model.eval()
        print('-> TTA disabled, using original model')

    # Load GT depths
    gt_path = os.path.join(splits_dir, args.eval_split, 'gt_depths.npz')
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)['data']

    pred_depths = []
    idx = 0

    for batch in dataloader:
        images = batch[("color", 0, 0)].cuda().float()
        bs = images.shape[0]
        batch_gt = gt_depths[idx:idx + bs]
        idx += bs

        sparse_depth, _ = create_sparse_depth(
            batch_gt,
            keep_ratio=args.sparse_keep_ratio,
            device=images.device,
        )

        if args.disable_tta:
            with torch.no_grad():
                if images.max() > 1.0:
                    images = images / 255.0
                mean = torch.tensor(args.tta_norm_mean).view(1, 3, 1, 1).to(images.device)
                std = torch.tensor(args.tta_norm_std).view(1, 3, 1, 1).to(images.device)
                images_norm = (images - mean) / std
                disp = depth_model(images_norm)
        else:
            disp = tta.adapt(images, sparse_depth)

        pred = disp.detach().cpu().numpy()
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
    print("\nCompletion-TTA Fast on SQLDepth (KITTI-C)")
    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")


if __name__ == '__main__':
    main()
