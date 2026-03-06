"""
VecTTAFast 测试脚本 - Kitti数据集

使用 Vectta.py 中的 VecTTAFast 类进行测试时适应。
VecTTAFast 是 VecTTA 的 Proxy-fast 变体，冻结所有模型参数，使用代理参数进行适应。
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import warnings

# 过滤警告
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")

from layers import disp_to_depth
from utils import readlines
import networks
from Vectta import SQLDepthModel, VecTTAFast

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
    parser.add_argument('--mode', type=str, choices=['generate_clean', 'adapt'], required=True, 
                       help='generate_clean:生成clean伪标签; adapt:自适应测试')
    parser.add_argument('--data_path', type=str, required=True, help='Kitti数据集路径')
    parser.add_argument('--load_weights_folder', type=str, required=True, help='模型权重文件夹路径')
    parser.add_argument('--eval_split', type=str, default='eigen', help='评估数据集split')
    parser.add_argument('--backbone', type=str, default='resnet_lite', help='backbone类型')
    parser.add_argument('--height', type=int, default=192, help='图像高度')
    parser.add_argument('--width', type=int, default=640, help='图像宽度')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--num_layers', type=int, default=50, help='ResNet层数')
    parser.add_argument('--num_features', type=int, default=256, help='特征数')
    parser.add_argument('--model_dim', type=int, default=32, help='模型维度')
    parser.add_argument('--patch_size', type=int, default=16, help='patch大小')
    parser.add_argument('--dim_out', type=int, default=64, help='输出维度')
    parser.add_argument('--query_nums', type=int, default=64, help='query数量')
    parser.add_argument('--min_depth', type=float, default=0.001, help='最小深度')
    parser.add_argument('--max_depth', type=float, default=80.0, help='最大深度')
    parser.add_argument('--disable_median_scaling', action='store_true', help='禁用中位数缩放')
    parser.add_argument('--pred_depth_scale_factor', type=float, default=1.0, help='深度缩放因子')
    parser.add_argument('--clean_pred_path', type=str, default='clean_pred_disps.npy', 
                       help='clean伪标签保存/加载路径')
    
    # VecTTAFast 参数
    parser.add_argument('--vec_steps', type=int, default=5, help='VecTTAFast 适应步数')
    parser.add_argument('--vec_update_mode', type=str, 
                       choices=['bn_only', 'bn_decoder', 'last_layers', 'all'], 
                       default='bn_only', help='更新模式')
    parser.add_argument('--vec_lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--vec_early_stop', action='store_true', help='启用早停机制')
    parser.add_argument('--vec_early_stop_patience', type=int, default=3, help='早停耐心值')
    parser.add_argument('--vec_grad_clip', type=float, default=1.0, help='梯度裁剪阈值')
    
    args = parser.parse_args()

    # === 数据集 ===
    splits_dir = os.path.join(os.path.dirname(__file__), "splits")
    filenames = readlines(os.path.join(splits_dir, args.eval_split, "test_files.txt"))
    from datasets.kitti_dataset import KITTIRAWDataset
    dataset = KITTIRAWDataset(
        args.data_path, filenames,
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

    # === 加载encoder/decoder和权重 ===
    if args.backbone in ["resnet", "resnet_lite"]:
        encoder = networks.ResnetEncoderDecoder(
            num_layers=args.num_layers,
            num_features=args.num_features,
            model_dim=args.model_dim
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
            decoder_channels=[1024, 512, 256, 128]
        )

    encoder_path = os.path.join(args.load_weights_folder, "encoder.pth")
    encoder_dict = torch.load(encoder_path)
    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
    encoder.cuda().eval()

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
    decoder.cuda().eval()
    
    model = SQLDepthModel(encoder, decoder)

    if args.mode == 'generate_clean':
        # === 生成clean伪标签 ===
        print("-> Generating clean predictions (no TTA)")
        model.eval()
        pred_disps = []
        with torch.no_grad():
            for batch in dataloader:
                images = batch[("color", 0, 0)].cuda()
                pred_disp = model(images)
                pred_disp = pred_disp.cpu().numpy()
                if pred_disp.ndim == 4:
                    pred_disp = pred_disp[:, 0, :, :]
                pred_disps.append(pred_disp)
        pred_disps = np.concatenate(pred_disps, axis=0)
        np.save(args.clean_pred_path, pred_disps)
        print(f"-> Clean伪标签已保存到 {args.clean_pred_path}")
        print(f"-> Shape: {pred_disps.shape}")
        return

    elif args.mode == 'adapt':
        # === 自适应推理 ===
        print("-> Loading clean predictions for VecTTAFast adaptation")
        clean_pred_disps = np.load(args.clean_pred_path)
        print(f"-> Loaded clean predictions from {args.clean_pred_path}")

        # 创建 VecTTAFast 适配器
        adapt_model = VecTTAFast(
            depth_model=model,
            update_mode=args.vec_update_mode,
            optimizer_lr=args.vec_lr,
            steps=args.vec_steps,
            early_stop=args.vec_early_stop,
            early_stop_patience=args.vec_early_stop_patience,
            grad_clip=args.vec_grad_clip,
        )
        adapt_model.cuda()
        print(f"-> VecTTAFast enabled with update_mode={args.vec_update_mode}, steps={args.vec_steps}, lr={args.vec_lr}")

        pred_disps = []
        idx = 0

        for batch in dataloader:
            images = batch[("color", 0, 0)].cuda()
            K = batch[("K", 0)].cuda()
            
            # 准备 clean_depth
            clean_depth = None
            if clean_pred_disps is not None:
                clean_disp_batch = clean_pred_disps[idx:idx+images.shape[0]]
                # 确保形状是 [B, H, W] 或 [B, 1, H, W]
                if clean_disp_batch.ndim == 2:
                    clean_disp_batch = clean_disp_batch[np.newaxis, ...]
                if clean_disp_batch.ndim == 3:
                    clean_disp_batch = clean_disp_batch[:, np.newaxis, ...]
                clean_depth = torch.from_numpy(clean_disp_batch).float().cuda()
                idx += images.shape[0]

            # VecTTAFast 适应和预测
            pred_disp = adapt_model(images, K, clean_depth=clean_depth)
            pred_disp = pred_disp.cpu().numpy()
            if pred_disp.ndim == 4:
                pred_disp = pred_disp[:, 0, :, :]
            pred_disps.append(pred_disp)
            
        pred_disps = np.concatenate(pred_disps, axis=0)

        # === 评测 ===
        gt_path = os.path.join(splits_dir, args.eval_split, "gt_depths.npz")
        gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]
        errors = []
        ratios = []
        MIN_DEPTH = args.min_depth
        MAX_DEPTH = args.max_depth

        print("-> Evaluating on Kitti")
        for i in range(pred_disps.shape[0]):
            gt_depth = gt_depths[i]
            gt_height, gt_width = gt_depth.shape[:2]
            pred_disp = pred_disps[i]
            pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
            pred_depth = pred_disp

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
        print("\nVecTTAFast on SQLDepth (Kitti)")
        print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
        print("\n-> Done!")

if __name__ == "__main__":
    main()
