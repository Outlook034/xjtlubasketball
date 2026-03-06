"""
为 Cityscapes 数据集生成 SQLdepth encoder 的 SSA 统计量

基于 ssa_feature_stats.py，适配 Cityscapes 数据集
"""

import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import networks
from ssa_utils import FeatureProjector
from utils import readlines

# 使得可以导入上级目录的 datasets
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import datasets


def parse_args():
    parser = argparse.ArgumentParser(
        description="生成 SQLdepth encoder 的 SSA 统计量（Cityscapes 版本）")
    parser.add_argument("--data_path", type=str, default=None,
                       help="Cityscapes 数据根目录")
    parser.add_argument("--eval_data_path", type=str, default=None,
                       help="Cityscapes 数据根目录（与 data_path 等价）")
    parser.add_argument("--eval_split", type=str, default="cityscapes")
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--model_dim", type=int, default=64)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--load_weights_folder", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="ssa_feature_stats_cityscapes.pt")
    parser.add_argument("--feature_pool", type=str, choices=["avg", "max"], default="avg")
    parser.add_argument("--max_samples", type=int, default=0,
                        help=">0 时仅使用指定数量的样本")
    return parser.parse_args()


def build_encoder(args):
    """构建 encoder（与 cityscapes_ssaTTA.py 一致）"""
    if args.backbone == "resnet50":
        encoder = networks.Resnet50EncoderDecoder(model_dim=args.model_dim)
    elif args.backbone in ["resnet", "resnet_lite"]:
        encoder = networks.ResnetEncoderDecoder(
            num_layers=50,
            num_features=256,
            model_dim=args.model_dim
        )
    elif args.backbone == "resnet18_lite":
        encoder = networks.LiteResnetEncoderDecoder(model_dim=args.model_dim)
    elif args.backbone == "eff_b5":
        encoder = networks.BaseEncoder.build(
            num_features=256, model_dim=args.model_dim)
    else:
        encoder = networks.Unet(
            pretrained=False,
            backbone=args.backbone,
            in_channels=3,
            num_classes=args.model_dim,
            decoder_channels=[1024, 512, 256, 128]
        )

    encoder_path = os.path.join(args.load_weights_folder, "encoder.pth")
    encoder_dict = torch.load(encoder_path, map_location="cpu")
    
    # 尝试从 checkpoint 中读取 height 和 width
    try:
        HEIGHT, WIDTH = encoder_dict['height'], encoder_dict['width']
        print(f"Using height={HEIGHT}, width={WIDTH} from checkpoint")
    except KeyError:
        HEIGHT, WIDTH = args.height, args.width
        print(f"Using height={HEIGHT}, width={WIDTH} from command line")
    
    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
    return encoder.cuda().eval(), HEIGHT, WIDTH


def main():
    args = parse_args()
    
    # 统一处理 data_path 和 eval_data_path
    if args.eval_data_path is not None:
        args.data_path = args.eval_data_path
    if args.data_path is None:
        raise ValueError("必须提供 --data_path 或 --eval_data_path 参数")
    
    splits_dir = os.path.join(os.path.dirname(__file__), "splits")
    filenames = readlines(os.path.join(splits_dir, args.eval_split, "test_files.txt"))
    
    # 构建 encoder 并获取实际使用的 height 和 width
    encoder, HEIGHT, WIDTH = build_encoder(args)
    
    # 使用 Cityscapes 数据集
    dataset = datasets.CityscapesEvalDataset(
        args.data_path, filenames,
        HEIGHT, WIDTH,
        [0], 1, is_train=False
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4,
                            pin_memory=True, drop_last=False)

    projector = FeatureProjector(pool_type=args.feature_pool)

    all_features = []
    used = 0
    max_samples = args.max_samples if args.max_samples and args.max_samples > 0 else None

    print(f"-> Extracting features from Cityscapes dataset (size: {HEIGHT}x{WIDTH})")
    print(f"-> Model dim: {args.model_dim}, Feature pool: {args.feature_pool}")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            images = batch[("color", 0, 0)].cuda()
            feat_map = encoder(images)
            feats = projector(feat_map)
            all_features.append(feats.cpu())
            used += feats.shape[0]
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {used} samples...")
            
            if max_samples is not None and used >= max_samples:
                print(f"  Reached max_samples={max_samples}, stopping...")
                break

    features = torch.cat(all_features, dim=0)
    if max_samples is not None:
        features = features[:max_samples]

    print(f"-> Computing PCA statistics from {features.shape[0]} samples...")
    mean = features.mean(dim=0)
    features_c = features - mean
    cov = features_c.T @ features_c / (features.shape[0] - 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "mean": mean.cpu(),
        "basis": eigvecs.cpu(),
        "eigvals": eigvals.cpu()
    }, str(output_path))

    print(f"\n-> SSA 统计量已保存到 {output_path}")
    print(f"   样本数: {features.shape[0]}")
    print(f"   特征维度: {features.shape[1]}")
    print(f"   主成分数量: {eigvals.shape[0]}")


if __name__ == "__main__":
    main()

