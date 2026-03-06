import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import argparse
from copy import deepcopy

from layers import disp_to_depth
from utils import readlines, normalize_image
import networks

# 1. 组合 encoder+decoder 为一个整体模型，只返回主Tensor
class SQLDepthModel(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, x):
        out = self.decoder(self.encoder(x))
        return out[("disp", 0)]  # 只返回主Tensor

# 2. EATA自适应模块，支持clean伪标签loss
class EATA(nn.Module):
    def __init__(self, model, optimizer, steps=1):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.criterion = nn.L1Loss()
    def forward(self, x, clean_depth=None):
        self.model.train()
        for _ in range(self.steps):
            out = self.model(x)
            if clean_depth is not None:
                if clean_depth.dim() == 3:
                    clean_depth = clean_depth.unsqueeze(1)  # [B, 1, H, W]
                # 假设out和clean_depth shape一致
                loss = self.criterion(out, clean_depth)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
        self.model.eval()
        with torch.no_grad():
            return self.model(x)

        
'''更新decoder+bn层'''        
# def configure_model_for_eata(model):
#     # 只解冻decoder和BN层参数，其余全部冻结
#     for param in model.parameters():
#         param.requires_grad = False
#     # 解冻decoder参数
#     for param in model.decoder.parameters():
#         param.requires_grad = True
#     # 解冻所有BN层参数
#     for m in model.modules(): 
#         if isinstance(m, nn.BatchNorm2d):
#             m.requires_grad_(True)
#             m.track_running_stats = False
#             m.running_mean = None
#             m.running_var = None
#             for p in m.parameters():
#                 p.requires_grad = True
#     return model

        
# def collect_decoder_and_bn_params(model):
#     params = list(model.decoder.parameters())
#     for m in model.modules():   
#         if isinstance(m, nn.BatchNorm2d):
#             params += list(m.parameters())
#     return params
        
'''只更新BN层'''        
# def configure_model_for_eata(model):
#     # 只解冻BN层参数，其余全部冻结
#     for param in model.parameters():
#         param.requires_grad = False
#     for m in model.modules():
#         if isinstance(m, nn.BatchNorm2d):
#             m.requires_grad_(True)
#             m.track_running_stats = False
#             m.running_mean = None
#             m.running_var = None
#             for p in m.parameters():
#                 p.requires_grad = True
#     return model


# def collect_bn_params(model):
#     params = []
#     for m in model.modules():
#         if isinstance(m, nn.BatchNorm2d):
#             params += list(m.parameters())
#     return params        
        
'''整个模型'''      
def configure_model_for_eata(model):
    for param in model.parameters():
        param.requires_grad = True
    return model




def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
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
    parser.add_argument('--mode', type=str, choices=['generate_clean', 'adapt'], required=True, help='generate_clean:生成clean伪标签; adapt:EATA自适应')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--log_dir', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='kitti')
    parser.add_argument('--eval_split', type=str, default='eigen')
    parser.add_argument('--backbone', type=str, default='resnet_lite')
    parser.add_argument('--height', type=int, default=192)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_layers', type=int, default=50)
    parser.add_argument('--num_features', type=int, default=256)
    parser.add_argument('--model_dim', type=int, default=32)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--dim_out', type=int, default=64)
    parser.add_argument('--query_nums', type=int, default=64)
    parser.add_argument('--min_depth', type=float, default=0.001)
    parser.add_argument('--max_depth', type=float, default=80.0)
    parser.add_argument('--load_weights_folder', type=str, required=True)
    parser.add_argument('--post_process', action='store_true')
    parser.add_argument('--disable_median_scaling', action='store_true')
    parser.add_argument('--pred_depth_scale_factor', type=float, default=1.0)
    parser.add_argument('--eata_steps', type=int, default=1)
    
    parser.add_argument('--clean_pred_path', type=str, default='clean_pred_disps.npy', help='clean伪标签保存/加载路径')
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
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

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
        encoder = networks.Unet(pretrained=False, backbone=args.backbone, in_channels=3, num_classes=args.model_dim, decoder_channels=[1024, 512, 256, 128])

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

    
    
    '''整个模型'''
    model = SQLDepthModel(encoder, decoder)
    configure_model_for_eata(model)

    if args.mode == 'generate_clean':
        # === 生成clean伪标签 ===
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
        print(f"Clean伪标签已保存到 {args.clean_pred_path}")
        return

    elif args.mode == 'adapt':
        # === EATA自适应推理 ===
        # 加载clean伪标签
        '''整个模型'''
        clean_pred_disps = np.load(args.clean_pred_path)
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=4e-5)
        adapt_model = EATA(model, optimizer, steps=args.eata_steps)
        '''BN层'''
        # clean_pred_disps = np.load(args.clean_pred_path)
        # bn_params = collect_bn_params(model)
        # optimizer = torch.optim.Adam(bn_params, lr=4e-5)
        # adapt_model = EATA(model, optimizer, steps=args.eata_steps)
        '''decoder+BN层'''
        # clean_pred_disps = np.load(args.clean_pred_path)
        # params = collect_decoder_and_bn_params(model)
        # optimizer = torch.optim.Adam(params, lr=4e-5)
        # adapt_model = EATA(model, optimizer, steps=args.eata_steps)
        
        pred_disps = []
        idx = 0
        for batch in dataloader:
            images = batch[("color", 0, 0)].cuda()
            # 取对应clean伪标签
            clean_depth = torch.from_numpy(clean_pred_disps[idx:idx+images.shape[0]]).cuda()
            idx += images.shape[0]
            pred_disp = adapt_model(images, clean_depth=clean_depth)
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
        print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
        print("\n-> Done!")

if __name__ == "__main__":
    main() 