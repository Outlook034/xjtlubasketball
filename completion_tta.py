import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from utils import readlines
import networks

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TTA_SRC = PROJECT_ROOT / "TTA-depth-completion-main" / "src"
if not TTA_SRC.exists():
    raise FileNotFoundError(f"TTA src path not found: {TTA_SRC}")
if str(TTA_SRC) not in sys.path:
    sys.path.append(str(TTA_SRC))

from loss_utils import sparse_depth_consistency_loss_func, smoothness_loss_func  # noqa: E402
from transforms import Transforms  # noqa: E402
from net_utils import OutlierRemoval  # noqa: E402


class SQLDepthModel(nn.Module):
    """Encoder + Decoder wrapper that only returns main disparity tensor."""

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        out = self.decoder(self.encoder(x))
        if isinstance(out, dict):
            return out[("disp", 0)]
        return out


class CompletionTTAModule(nn.Module):
    """Wrapper that plugs TTA-depth-completion style adaptation into SQLDepth."""

    def __init__(self, depth_model, args):
        super().__init__()
        self.depth_model = depth_model
        self.steps = args.tta_steps
        self.lr = args.tta_lr
        self.w_sparse = args.tta_w_sparse
        self.w_smooth = args.tta_w_smooth
        self.grad_clip = args.tta_grad_clip
        self.augmentation_prob = args.tta_aug_prob

        self.outlier_removal = OutlierRemoval(7, 1.5)
        self.train_geom = Transforms(
            random_crop_to_shape=[args.height, args.width],
            random_flip_type=args.tta_aug_flip_type,
            random_rotate_max=args.tta_aug_rotate,
            random_crop_and_pad=[-1, -1],
            random_resize_and_pad=[-1, -1],
            random_resize_and_crop=[-1, -1]
        )
        self.interpolation_modes = self.train_geom.map_interpolation_mode_names_to_enums(
            ['bilinear', 'nearest', 'nearest']
        )
        self.train_photo = Transforms(
            normalized_image_range=[0, 255],
            random_brightness=args.tta_aug_brightness,
            random_contrast=args.tta_aug_contrast,
            random_gamma=args.tta_aug_gamma,
            random_hue=[-1, -1],
            random_saturation=args.tta_aug_saturation,
            random_noise_type=args.tta_aug_noise_type,
            random_noise_spread=args.tta_aug_noise_spread
        )

        self.configure_params(args.tta_update_mode)
        self.optimizer = torch.optim.Adam(self.params_to_update, lr=self.lr)
        self.register_buffer("norm_mean", torch.tensor(args.tta_norm_mean).view(1, 3, 1, 1))
        self.register_buffer("norm_std", torch.tensor(args.tta_norm_std).view(1, 3, 1, 1))

    def configure_params(self, mode):
        for p in self.depth_model.parameters():
            p.requires_grad = False
        params = []

        if mode == "all":
            for p in self.depth_model.parameters():
                p.requires_grad = True
                params.append(p)
        elif mode in ["bn", "bn_decoder"]:
            for m in self.depth_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.requires_grad_(True)
                    m.track_running_stats = False
                    m.running_mean = None
                    m.running_var = None
                    params += list(m.parameters())
            if mode == "bn_decoder" and hasattr(self.depth_model, "decoder"):
                for p in self.depth_model.decoder.parameters():
                    if not p.requires_grad:
                        p.requires_grad = True
                    params.append(p)
        else:
            raise ValueError(f"Unsupported update mode: {mode}")

        if not params:
            raise ValueError(f"No parameters configured for update mode '{mode}'.")
        self.params_to_update = torch.nn.ParameterList(params)

    def _normalize(self, image):
        if image.max() > 1.0:
            image = image / 255.0
        return (image - self.norm_mean) / self.norm_std

    @torch.enable_grad()
    def adapt(self, image, sparse_depth):
        validity = (sparse_depth > 0).float()
        filtered_sparse, filtered_valid = self.outlier_removal.remove_outliers(
            sparse_depth=sparse_depth,
            validity_map=validity
        )

        geom_outputs = self.train_geom.transform(
            images_arr=[image, filtered_sparse, filtered_valid],
            intrinsics_arr=[],
            interpolation_modes=self.interpolation_modes,
            random_transform_probability=self.augmentation_prob
        )
        image_geom, sparse_geom, valid_geom = geom_outputs

        # Resize sparse/valid maps to match geometric output size
        sparse_geom = torch.nn.functional.interpolate(
            sparse_geom,
            size=image_geom.shape[2:],
            mode='nearest'
        )
        valid_geom = torch.nn.functional.interpolate(
            valid_geom,
            size=image_geom.shape[2:],
            mode='nearest'
        )

        [image_photo] = self.train_photo.transform(
            images_arr=[image_geom],
            random_transform_probability=self.augmentation_prob
        )

        for _ in range(self.steps):
            norm_image = self._normalize(image_photo)
            prediction = self.depth_model(norm_image)
            if isinstance(prediction, dict):
                prediction = prediction[("disp", 0)]

            sparse_for_loss = sparse_geom
            valid_for_loss = valid_geom
            if sparse_for_loss.shape[2:] != prediction.shape[2:]:
                sparse_for_loss = torch.nn.functional.interpolate(
                    sparse_for_loss,
                    size=prediction.shape[2:],
                    mode='nearest'
                )
                valid_for_loss = torch.nn.functional.interpolate(
                    valid_for_loss,
                    size=prediction.shape[2:],
                    mode='nearest'
                )
            image_for_loss = image_photo
            if image_for_loss.shape[2:] != prediction.shape[2:]:
                image_for_loss = torch.nn.functional.interpolate(
                    image_for_loss,
                    size=prediction.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )

            loss_sparse = sparse_depth_consistency_loss_func(
                prediction,
                sparse_for_loss,
                valid_for_loss
            )
            loss_smooth = smoothness_loss_func(
                predict=prediction,
                image=image_for_loss / 255.0 if image_for_loss.max() > 1.0 else image_for_loss
            )
            total_loss = self.w_sparse * loss_sparse + self.w_smooth * loss_smooth

            self.optimizer.zero_grad()
            total_loss.backward()
            if self.grad_clip is not None and len(self.params_to_update) > 0:
                torch.nn.utils.clip_grad_norm_(self.params_to_update, self.grad_clip)
            self.optimizer.step()

        self.depth_model.eval()
        with torch.no_grad():
            final_prediction = self.depth_model(self._normalize(image))
            if isinstance(final_prediction, dict):
                final_prediction = final_prediction[("disp", 0)]
        return final_prediction


def create_sparse_depth(gt_depth, keep_ratio=0.1, device="cuda"):
    """Sample sparse LiDAR points from dense ground truth batch."""
    gt = torch.from_numpy(gt_depth).float().to(device)
    mask = (gt > 0).float()
    if keep_ratio < 1.0:
        rand = torch.rand_like(gt)
        mask = mask * (rand < keep_ratio).float()
    sparse_depth = gt * mask
    return sparse_depth.unsqueeze(1), mask.unsqueeze(1)


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    rmse = np.sqrt(((gt - pred) ** 2).mean())
    rmse_log = np.sqrt(((np.log(gt) - np.log(pred)) ** 2).mean())
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)
    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def load_sql_depth(args):
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
            pretrained=False,
            backbone=args.backbone,
            in_channels=3,
            num_classes=args.model_dim,
            decoder_channels=[1024, 512, 256, 128]
        )

    encoder_path = os.path.join(args.load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(args.load_weights_folder, "depth.pth")

    encoder_dict = torch.load(encoder_path, map_location="cpu")
    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})

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
    decoder.load_state_dict(torch.load(decoder_path, map_location="cpu"))

    model = SQLDepthModel(encoder, decoder).cuda()
    return model


def main():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('--eval_data_path', type=str, required=True, help='Cityscapes dataset root path')
    parser.add_argument('--eval_split', type=str, default='cityscapes')
    parser.add_argument('--load_weights_folder', type=str, required=True)
    parser.add_argument('--backbone', type=str, default='resnet_lite')
    parser.add_argument('--height', type=int, default=320)
    parser.add_argument('--width', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=4)
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

    # Completion-TTA hyper-params
    parser.add_argument('--tta_steps', type=int, default=1)
    parser.add_argument('--tta_lr', type=float, default=4e-5)
    parser.add_argument('--tta_update_mode', type=str, choices=['bn', 'bn_decoder', 'all'], default='bn_decoder')
    parser.add_argument('--tta_w_sparse', type=float, default=1.0)
    parser.add_argument('--tta_w_smooth', type=float, default=1.0)
    parser.add_argument('--tta_grad_clip', type=float, default=5.0)
    parser.add_argument('--sparse_keep_ratio', type=float, default=0.1)
    parser.add_argument('--tta_norm_mean', nargs='+', type=float, default=[0.485, 0.456, 0.406])
    parser.add_argument('--tta_norm_std', nargs='+', type=float, default=[0.229, 0.224, 0.225])
    parser.add_argument('--tta_aug_prob', type=float, default=0.5)
    parser.add_argument('--tta_aug_crop_type', nargs='+', type=str, default=['none'])
    parser.add_argument('--tta_aug_flip_type', nargs='+', type=str, default=['none'])
    parser.add_argument('--tta_aug_rotate', type=float, default=0.0)
    parser.add_argument('--tta_aug_brightness', nargs='+', type=float, default=[-1, -1])
    parser.add_argument('--tta_aug_contrast', nargs='+', type=float, default=[-1, -1])
    parser.add_argument('--tta_aug_gamma', nargs='+', type=float, default=[-1, -1])
    parser.add_argument('--tta_aug_saturation', nargs='+', type=float, default=[-1, -1])
    parser.add_argument('--tta_aug_noise_type', type=str, default='none')
    parser.add_argument('--tta_aug_noise_spread', type=float, default=-1)

    args = parser.parse_args()

    splits_dir = os.path.join(os.path.dirname(__file__), "splits")
    filenames = readlines(os.path.join(splits_dir, args.eval_split, "test_files.txt"))
    if args.eval_split == "cityscapes":
        from datasets.cityscapes_evaldataset import CityscapesEvalDataset
        dataset = CityscapesEvalDataset(
            args.eval_data_path, filenames,
            args.height, args.width,
            [0], 1, is_train=False
        )
    else:
        from datasets.kitti_dataset import KITTIRAWDataset
        dataset = KITTIRAWDataset(
            args.eval_data_path, filenames,
            args.height, args.width,
            [0], 1, is_train=False
        )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )

    model = load_sql_depth(args)
    completion_tta = CompletionTTAModule(model, args).cuda()

    # Load GT depths for evaluation and sparse sampling
    if args.eval_split == "cityscapes":
        gt_depths_dir = os.path.join(splits_dir, args.eval_split, "gt_depths")
    else:
        gt_path = os.path.join(splits_dir, args.eval_split, "gt_depths.npz")
        gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

    pred_depths = []
    idx = 0
    for batch in dataloader:
        images = batch[("color", 0, 0)].cuda().float()
        bs = images.shape[0]
        if args.eval_split == "cityscapes":
            batch_gt_list = []
            for b in range(bs):
                gt_depth = np.load(os.path.join(gt_depths_dir, str(idx + b).zfill(3) + '_depth.npy'))
                # crop ground truth to remove ego car (match dataloader behavior)
                gt_height, _ = gt_depth.shape[:2]
                gt_height = int(round(gt_height * 0.75))
                gt_depth = gt_depth[:gt_height]
                batch_gt_list.append(gt_depth)
            batch_gt = np.stack(batch_gt_list, axis=0)
            idx += bs
        else:
            batch_gt = gt_depths[idx:idx + bs]
            idx += bs

        sparse_depth, validity = create_sparse_depth(
            batch_gt,
            keep_ratio=args.sparse_keep_ratio,
            device=images.device
        )

        pred = completion_tta.adapt(images, sparse_depth)
        pred = pred.detach().cpu().numpy()
        if pred.ndim == 4:
            pred = pred[:, 0, :, :]
        pred_depths.append(pred)

    pred_depths = np.concatenate(pred_depths, axis=0)

    # Evaluation (same as VECTTA)
    errors = []
    ratios = []
    MIN_DEPTH = args.min_depth
    MAX_DEPTH = args.max_depth

    for i in range(pred_depths.shape[0]):
        if args.eval_split == "cityscapes":
            gt_depth = np.load(os.path.join(gt_depths_dir, str(i).zfill(3) + '_depth.npy'))
            gt_height, gt_width = gt_depth.shape[:2]
            gt_height = int(round(gt_height * 0.75))
            gt_depth = gt_depth[:gt_height]
        else:
            gt_depth = gt_depths[i]
            gt_height, gt_width = gt_depth.shape[:2]

        pred_depth = pred_depths[i]
        pred_depth = cv2.resize(pred_depth, (gt_width, gt_height))

        if args.eval_split == "cityscapes":
            # centre crop to middle 50% (bottom 25% already removed)
            gt_depth = gt_depth[256:, 192:1856]
            pred_depth = pred_depth[256:, 192:1856]
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
        else:
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width, 0.96405229 * gt_width]).astype(np.int32)
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
    print("\nCompletion-TTA on SQLDepth")
    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")


if __name__ == "__main__":
    main()

