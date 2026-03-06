import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import networks
from datasets.kitti_dataset import KITTIRAWDataset
from ssa_utils import FeatureProjector
from utils import readlines


def parse_args():
    parser = argparse.ArgumentParser(
        description="生成 SQLdepth encoder 的 SSA 统计量")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--eval_split", type=str, default="eigen")
    parser.add_argument("--backbone", type=str, default="resnet_lite")
    parser.add_argument("--num_layers", type=int, default=50)
    parser.add_argument("--num_features", type=int, default=256)
    parser.add_argument("--model_dim", type=int, default=32)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--dim_out", type=int, default=64)
    parser.add_argument("--query_nums", type=int, default=64)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--load_weights_folder", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="ssa_feature_stats.pt")
    parser.add_argument("--feature_pool", type=str, choices=["avg", "max"], default="avg")
    parser.add_argument("--max_samples", type=int, default=0,
                        help=">0 时仅使用指定数量的样本")
    return parser.parse_args()


def build_encoder(args):
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
            num_features=args.num_features, model_dim=args.model_dim)
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
    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
    return encoder.cuda().eval()


def main():
    args = parse_args()
    splits_dir = os.path.join(os.path.dirname(__file__), "splits")
    filenames = readlines(os.path.join(splits_dir, args.eval_split, "test_files.txt"))
    dataset = KITTIRAWDataset(
        args.data_path, filenames,
        args.height, args.width,
        [0], 1, is_train=False
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4,
                            pin_memory=True, drop_last=False)

    encoder = build_encoder(args)
    projector = FeatureProjector(pool_type=args.feature_pool)

    all_features = []
    used = 0
    max_samples = args.max_samples if args.max_samples and args.max_samples > 0 else None

    with torch.no_grad():
        for batch in dataloader:
            images = batch[("color", 0, 0)].cuda()
            feat_map = encoder(images)
            feats = projector(feat_map)
            all_features.append(feats.cpu())
            used += feats.shape[0]
            if max_samples is not None and used >= max_samples:
                break

    features = torch.cat(all_features, dim=0)
    if max_samples is not None:
        features = features[:max_samples]

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

    print(f"SSA 统计量已保存到 {output_path}, 样本数: {features.shape[0]}")


if __name__ == "__main__":
    main()

