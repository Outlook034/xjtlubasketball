"""
Tent-style (regression) for Depth Estimation on Cityscapes

- Uses TentRegression from Vectta.py (BN-only + flip consistency)
- No clean pseudo-labels required
- Evaluation protocol aligned with existing Cityscapes scripts
"""

import os
import sys
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
from copy import deepcopy

from utils import readlines
from layers import disp_to_depth
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import datasets
import networks

# Import TentRegression from Vectta.py
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
from Vectta import SQLDepthModel

class TentRegressionCityscapes(nn.Module):
    def __init__(
        self,
        model,
        lr=1e-4,
        steps=1,
        episodic=False,
        use_flip=False,
        w_consistency=1.0,
        grad_clip=1.0,
    ):
        super().__init__()
        self.model = model
        self.steps = steps
        self.episodic = episodic
        self.use_flip = use_flip
        self.w_consistency = w_consistency
        self.grad_clip = grad_clip

        self.model_state = deepcopy(self.model.state_dict())

        configure_model(self.model)
        params, _ = collect_params(self.model)
        self.optimizer = torch.optim.Adam(params, lr=lr)

    def reset(self):
        self.model.load_state_dict(self.model_state, strict=True)

    def _forward_model(self, x):
        out = self.model(x)
        if isinstance(out, dict):
            out = out[("disp", 0)]
        return out

    def forward(self, x, clean_depth=None):
        if self.episodic:
            self.reset()

        self.model.train()

        for _ in range(self.steps):
            pred = self._forward_model(x)

            loss = pred.new_tensor(0.0)

            if clean_depth is not None:
                if clean_depth.dim() == 3:
                    clean_depth = clean_depth.unsqueeze(1)
                if clean_depth.shape[2:] != pred.shape[2:]:
                    clean_depth = F.interpolate(clean_depth, size=pred.shape[2:], mode='bilinear', align_corners=False)
                loss = loss + F.l1_loss(pred, clean_depth)

            if self.use_flip:
                x_flip = torch.flip(x, dims=[3])
                pred_flip = self._forward_model(x_flip)
                pred_flip_back = torch.flip(pred_flip, dims=[3])
                loss = loss + self.w_consistency * F.l1_loss(pred, pred_flip_back)

            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

        self.model.eval()
        with torch.no_grad():
            return self._forward_model(x)

cv2.setNumThreads(0)


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


def main():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('--eval_data_path', type=str, required=True, help='Cityscapes dataset path')
    parser.add_argument('--eval_split', type=str, default='cityscapes', help='Eval split')
    parser.add_argument('--load_weights_folder', type=str, required=True, help='Model weights folder')
    parser.add_argument('--backbone', type=str, default='resnet50', help='Backbone type')
    parser.add_argument('--height', type=int, default=192, help='Image height')
    parser.add_argument('--width', type=int, default=640, help='Image width')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--model_dim', type=int, default=32, help='Model dimension')
    parser.add_argument('--patch_size', type=int, default=16, help='Patch size')
    parser.add_argument('--dim_out', type=int, default=64, help='Output dimension')
    parser.add_argument('--query_nums', type=int, default=64, help='Query number')
    parser.add_argument('--min_depth', type=float, default=1e-3, help='Min depth')
    parser.add_argument('--max_depth', type=float, default=80.0, help='Max depth')
    parser.add_argument('--disable_median_scaling', action='store_true', help='Disable median scaling')
    parser.add_argument('--pred_depth_scale_factor', type=float, default=1.0, help='Depth scale factor')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers')
    parser.add_argument('--disable_tta', action='store_true', help='Disable TTA (baseline)')

    # Tent-style params
    parser.add_argument('--tent_steps', type=int, default=1, help='Tent adaptation steps')
    parser.add_argument('--tent_lr', type=float, default=1e-4, help='Tent learning rate')
    parser.add_argument('--tent_episodic', action='store_true', help='Enable episodic reset')
    parser.add_argument('--tent_flip', action='store_true', default=True, help='Use flip consistency')
    parser.add_argument('--tent_w_consistency', type=float, default=1.0, help='Flip consistency weight')
    parser.add_argument('--tent_grad_clip', type=float, default=1.0, help='Gradient clipping')

    args = parser.parse_args()

    # === Dataset ===
    splits_dir = os.path.join(os.path.dirname(__file__), "splits")
    filenames = readlines(os.path.join(splits_dir, args.eval_split, "test_files.txt"))
    dataset = datasets.CityscapesEvalDataset(
        args.eval_data_path,
        filenames,
        args.height,
        args.width,
        frame_idxs=[0],
        num_scales=4,
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

    # === Load model ===
    encoder_path = os.path.join(args.load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(args.load_weights_folder, "depth.pth")
    encoder_dict = torch.load(encoder_path, map_location='cpu')

    try:
        HEIGHT, WIDTH = encoder_dict['height'], encoder_dict['width']
    except KeyError:
        HEIGHT, WIDTH = args.height, args.width

    if args.backbone == 'resnet50':
        encoder = networks.ResnetEncoderDecoder(num_layers=50, num_features=512, model_dim=args.model_dim)
    else:
        raise NotImplementedError(f"Backbone {args.backbone} not supported in this script")

    # Cityscapes models use ResnetEncoderDecoder + Depth_Decoder_QueryTr (transformer)
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

    # Load encoder weights (strip 'encoder.' prefix if present)
    encoder_state = {k.replace('encoder.', ''): v for k, v in encoder_dict.items() if k.startswith('encoder.')}
    encoder.load_state_dict(encoder_state, strict=False)

    # Load decoder weights
    decoder_dict = torch.load(decoder_path, map_location='cpu')
    decoder.load_state_dict(decoder_dict, strict=False)

    encoder.cuda().eval()
    decoder.cuda().eval()
    depth_model = SQLDepthModel(encoder, decoder).cuda()

    if not args.disable_tta:
        tent = TentRegressionCityscapes(
            model=depth_model,
            lr=args.tent_lr,
            steps=args.tent_steps,
            episodic=args.tent_episodic,
            use_flip=args.tent_flip,
            w_consistency=args.tent_w_consistency,
            grad_clip=args.tent_grad_clip,
        ).cuda()
        print(f"-> Tent enabled: steps={args.tent_steps}, lr={args.tent_lr}, flip={args.tent_flip}")
    else:
        tent = None
        depth_model.eval()
        print('-> TTA disabled, using original model')

    # === Evaluation ===
    pred_disps = []

    mean = torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1, 3, 1, 1)

    for batch in tqdm(dataloader, desc="Evaluating Cityscapes with Tent"):
        images = batch[("color", 0, 0)].cuda().float()
        if images.max() > 1.0:
            images = images / 255.0
        images_norm = (images - mean) / std

        if not args.disable_tta:
            disp_out = tent(images_norm)
        else:
            with torch.no_grad():
                disp_out = depth_model(images_norm)

        pred = disp_out.detach().cpu().numpy()
        if pred.ndim == 4:
            pred = pred[:, 0, :, :]
        pred_disps.append(pred)

    pred_disps = np.concatenate(pred_disps, axis=0)

    # === Load GT and compute metrics ===
    # Cityscapes GT in this repo is stored as per-sample npy files:
    # splits/cityscapes/gt_depths/{i}_depth.npy
    gt_dir = os.path.join(splits_dir, args.eval_split, "gt_depths")
    if not os.path.isdir(gt_dir):
        gt_path = os.path.join(splits_dir, args.eval_split, "gt_depths.npz")
        if os.path.exists(gt_path):
            gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)['data']
        else:
            print(f"-> GT not found: {gt_dir} or {gt_path}")
            print("-> Skip metric computation.")
            print(f"-> Pred disp shape: {pred_disps.shape}")
            print("\n-> Done!")
            return
    else:
        gt_depths = None

    errors = []
    ratios = []
    MIN_DEPTH = args.min_depth
    MAX_DEPTH = args.max_depth

    for i in range(pred_disps.shape[0]):
        if gt_depths is not None:
            gt_depth = gt_depths[i]
        else:
            # Cityscapes scripts use zero-padded 3-digit naming (e.g., 000_depth.npy)
            gt_file = os.path.join(gt_dir, f"{i:03d}_depth.npy")
            if not os.path.exists(gt_file):
                raise FileNotFoundError(f"GT file not found for index {i}: tried {gt_file}")
            gt_depth = np.load(gt_file)

        gt_height, gt_width = gt_depth.shape[:2]

        # Remove bottom 25% (consistent with other Cityscapes scripts)
        gt_height = int(round(gt_height * 0.75))
        gt_depth = gt_depth[:gt_height]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = pred_disp

        # Centre crop to middle 50% (consistent with other Cityscapes scripts)
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
    print("\nTent-style on SQLDepth (Cityscapes)")
    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")


if __name__ == '__main__':
    main()
