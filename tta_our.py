import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from copy import deepcopy

from layers import disp_to_depth
from utils import readlines, normalize_image
import networks

# 1. 深度模型
class SQLDepthModel(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, x):
        out = self.decoder(self.encoder(x))
        return out[("disp", 0)]  # 只返回主Tensor

# 2. 位姿模型
class PoseModel(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, x1, x2):
        # 将两帧图像拼接为一个6通道输入
        x = torch.cat([x1, x2], dim=1)
        out = self.decoder(self.encoder(x))
        return out  # 返回相对位姿 [B, 6] (平移+旋转)

# 3. 自监督TTA适应模块
class SelfSupervisedTTA(nn.Module):
    def __init__(self, depth_model, pose_model, depth_optimizer, pose_optimizer, steps=1, 
                 lambda_photo=1.0, lambda_smooth=0.1, lambda_geo=0.5):
        super().__init__()
        self.depth_model = depth_model
        self.pose_model = pose_model
        self.depth_optimizer = depth_optimizer
        self.pose_optimizer = pose_optimizer
        self.steps = steps
        
        # 损失权重
        self.lambda_photo = lambda_photo
        self.lambda_smooth = lambda_smooth
        self.lambda_geo = lambda_geo
        
        # SSIM权重
        self.ssim_weight = 0.85
        
    def compute_reprojection_loss(self, pred, target):
        """计算光度重投影损失，结合SSIM和L1"""
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)
        
        # SSIM损失
        ssim_loss = self.compute_ssim(pred, target)
        ssim_loss = (1 - ssim_loss) / 2  # 转换为损失
        
        # 组合损失
        reprojection_loss = self.ssim_weight * ssim_loss + (1 - self.ssim_weight) * l1_loss
        return reprojection_loss
    
    def compute_ssim(self, x, y):
        """简化版SSIM计算"""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        mu_x = F.avg_pool2d(x, 3, 1, 1)
        mu_y = F.avg_pool2d(y, 3, 1, 1)
        
        sigma_x = F.avg_pool2d(x ** 2, 3, 1, 1) - mu_x ** 2
        sigma_y = F.avg_pool2d(y ** 2, 3, 1, 1) - mu_y ** 2
        sigma_xy = F.avg_pool2d(x * y, 3, 1, 1) - mu_x * mu_y
        
        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
        
        return torch.clamp((SSIM_n / SSIM_d), 0, 1)
    
    def compute_smoothness_loss(self, disp, img):
        """计算深度平滑损失"""
        grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
        grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])
        
        grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
        grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)
        
        grad_disp_x *= torch.exp(-grad_img_x)
        grad_disp_y *= torch.exp(-grad_img_y)
        
        return grad_disp_x.mean() + grad_disp_y.mean()
    
    def compute_geometric_consistency_loss(self, depth, warped_depth, valid_mask):
        """计算几何一致性损失"""
        diff = torch.abs(depth - warped_depth) * valid_mask
        return diff.sum() / (valid_mask.sum() + 1e-7)
    
    def warp_frame(self, img, depth, pose, intrinsics):
        """将source图像重投影到target视角"""
        batch_size, _, height, width = depth.shape
        
        # 生成像素网格
        meshgrid = torch.meshgrid(torch.linspace(0, width-1, width), 
                                   torch.linspace(0, height-1, height))
        id_coords = torch.stack(meshgrid, dim=0).to(depth.device)
        id_coords = id_coords.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        
        # 获取深度值
        depth = depth.squeeze(1)
        
        # 构建相机点
        ones = torch.ones_like(depth)
        pix_coords = torch.stack([id_coords[:, 0], id_coords[:, 1], ones], dim=1)
        
        # 反投影到3D空间
        inv_K = torch.inverse(intrinsics)
        cam_points = torch.matmul(inv_K[:, :3, :3], pix_coords.reshape(batch_size, 3, -1))
        cam_points = depth.reshape(batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, ones.reshape(batch_size, 1, -1)], dim=1)
        
        # 应用位姿变换
        T = pose_vec2mat(pose)
        P = torch.matmul(T, cam_points)
        
        # 重投影到新视角
        P = P[:, :3, :]
        P = torch.matmul(intrinsics[:, :3, :3], P)
        P = P.reshape(batch_size, 3, height, width)
        
        # 归一化像素坐标
        pix_coords = P[:, :2, :, :] / (P[:, 2:3, :, :] + 1e-7)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= width - 1
        pix_coords[..., 1] /= height - 1
        pix_coords = (pix_coords - 0.5) * 2  # [-1, 1]范围
        
        # 生成有效掩码
        valid_mask = (pix_coords[:, :, :, 0] > -1) & (pix_coords[:, :, :, 0] < 1) & \
                     (pix_coords[:, :, :, 1] > -1) & (pix_coords[:, :, :, 1] < 1)
        valid_mask = valid_mask.unsqueeze(1).float()
        
        # 采样重投影图像
        warped_img = F.grid_sample(img, pix_coords, padding_mode='zeros', align_corners=False)
        
        return warped_img, valid_mask
    
    def warp_depth(self, depth, target_depth, pose, intrinsics):
        """将source深度图重投影到target视角"""
        warped_img, valid_mask = self.warp_frame(depth, target_depth, pose, intrinsics)
        return warped_img, valid_mask
    
    def forward(self, target_frame, source_frames, intrinsics=None, clean_depth=None):
        """执行自监督测试时适应
        Args:
            target_frame: 目标帧 [B, 3, H, W]
            source_frames: 源帧列表 [[B, 3, H, W], ...]
            intrinsics: 相机内参 [B, 4, 4]
            clean_depth: 可选的伪标签深度 [B, H, W]
        """
        self.depth_model.train()
        self.pose_model.train()
        
        # 如果没有提供内参，使用默认值
        if intrinsics is None:
            batch_size, _, height, width = target_frame.shape
            intrinsics = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1).to(target_frame.device)
            intrinsics[:, 0, 0] = 0.58 * width  # fx
            intrinsics[:, 1, 1] = 0.58 * height  # fy
            intrinsics[:, 0, 2] = 0.5 * width  # cx
            intrinsics[:, 1, 2] = 0.5 * height  # cy
        
        for _ in range(self.steps):
            total_loss = 0
            
            # 预测目标帧深度
            target_disp = self.depth_model(target_frame)
            target_depth = disp_to_depth(target_disp, min_depth=0.1, max_depth=100)[1]
            
            # 伪标签损失（如果提供）
            if clean_depth is not None:
                if clean_depth.dim() == 3:
                    clean_depth = clean_depth.unsqueeze(1)
                # 假设clean_depth是视差而非深度
                clean_loss = F.l1_loss(target_disp, clean_depth)
                total_loss += clean_loss
            
            # 自监督损失
            photo_losses = []
            smooth_loss = self.compute_smoothness_loss(target_disp, target_frame)
            geo_losses = []
            
            for source_frame in source_frames:
                # 估计相对位姿
                pose = self.pose_model(target_frame, source_frame)
                
                # 图像重投影
                warped_source, valid_mask = self.warp_frame(source_frame, target_depth, pose, intrinsics)
                
                # 计算光度损失
                photo_loss = self.compute_reprojection_loss(warped_source, target_frame)
                photo_loss = (photo_loss * valid_mask).sum() / (valid_mask.sum() + 1e-7)
                photo_losses.append(photo_loss)
                
                # 计算几何一致性损失
                source_disp = self.depth_model(source_frame)
                source_depth = disp_to_depth(source_disp, min_depth=0.1, max_depth=100)[1]
                warped_source_depth, depth_valid_mask = self.warp_depth(source_depth, target_depth, pose, intrinsics)
                
                # 组合有效掩码
                combined_mask = valid_mask * depth_valid_mask
                
                geo_loss = self.compute_geometric_consistency_loss(target_depth, warped_source_depth, combined_mask)
                geo_losses.append(geo_loss)
            
            # 取最小的光度损失（处理遮挡）
            photo_loss = torch.min(torch.stack(photo_losses), dim=0)[0] if photo_losses else 0
            geo_loss = torch.mean(torch.stack(geo_losses)) if geo_losses else 0
            
            # 组合损失
            ss_loss = self.lambda_photo * photo_loss + \
                      self.lambda_smooth * smooth_loss + \
                      self.lambda_geo * geo_loss
            
            total_loss += ss_loss
            
            # 反向传播
            self.depth_optimizer.zero_grad()
            self.pose_optimizer.zero_grad()
            total_loss.backward()
            self.depth_optimizer.step()
            self.pose_optimizer.step()
        
        # 切换回评估模式并返回深度预测
        self.depth_model.eval()
        self.pose_model.eval()
        with torch.no_grad():
            return self.depth_model(target_frame)


# 辅助函数：将位姿向量转换为变换矩阵
def pose_vec2mat(vec):
    """将位姿向量 [B, 6] 转换为变换矩阵 [B, 4, 4]"""
    batch_size = vec.shape[0]
    translation = vec[:, :3].unsqueeze(-1)  # [B, 3, 1]
    rot = vec[:, 3:]  # [B, 3]
    
    # 旋转向量转换为旋转矩阵
    rot_mat = euler2mat(rot)  # [B, 3, 3]
    
    # 构建变换矩阵
    transform_mat = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1).to(vec.device)
    transform_mat[:, :3, :3] = rot_mat
    transform_mat[:, :3, 3:4] = translation
    
    return transform_mat

def euler2mat(angle):
    """欧拉角转旋转矩阵（简化版）"""
    B = angle.size(0)
    x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]
    
    cosz = torch.cos(z)
    sinz = torch.sin(z)
    cosy = torch.cos(y)
    siny = torch.sin(y)
    cosx = torch.cos(x)
    sinx = torch.sin(x)
    
    # 构建旋转矩阵
    r11 = cosy * cosz
    r12 = -cosy * sinz
    r13 = siny
    r21 = cosx * sinz + sinx * siny * cosz
    r22 = cosx * cosz - sinx * siny * sinz
    r23 = -sinx * cosy
    r31 = sinx * sinz - cosx * siny * cosz
    r32 = sinx * cosz + cosx * siny * sinz
    r33 = cosx * cosy
    
    matrix = torch.empty(B, 3, 3, device=angle.device)
    matrix[:, 0, 0] = r11
    matrix[:, 0, 1] = r12
    matrix[:, 0, 2] = r13
    matrix[:, 1, 0] = r21
    matrix[:, 1, 1] = r22
    matrix[:, 1, 2] = r23
    matrix[:, 2, 0] = r31
    matrix[:, 2, 1] = r32
    matrix[:, 2, 2] = r33
    
    return matrix


# 选择性参数更新配置
def configure_model_for_ss_tta(depth_model, pose_model):
    """配置模型进行自监督测试时适应"""
    # 冻结所有参数
    for param in depth_model.parameters():
        param.requires_grad = False
    for param in pose_model.parameters():
        param.requires_grad = False
    
    # 解冻深度模型的decoder和后期编码器层
    for param in depth_model.decoder.parameters():
        param.requires_grad = True
    
    # 解冻深度编码器的后期层(假设是ResNet)
    if hasattr(depth_model.encoder, 'layer3'):
        for param in depth_model.encoder.layer3.parameters():
            param.requires_grad = True
    if hasattr(depth_model.encoder, 'layer4'):
        for param in depth_model.encoder.layer4.parameters():
            param.requires_grad = True
    
    # 解冻位姿网络的后期层
    for param in pose_model.decoder.parameters():
        param.requires_grad = True
    
    # 特殊处理BN层
    for m in depth_model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
            for p in m.parameters():
                p.requires_grad = True
    
    for m in pose_model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
            for p in m.parameters():
                p.requires_grad = True
                
    return depth_model, pose_model


# 收集需要更新的参数
def collect_ss_tta_params(depth_model, pose_model):
    """收集自监督TTA需要更新的参数"""
    depth_params = []
    pose_params = []
    
    # 深度模型参数
    for name, param in depth_model.named_parameters():
        if param.requires_grad:
            depth_params.append(param)
    
    # 位姿模型参数
    for name, param in pose_model.named_parameters():
        if param.requires_grad:
            pose_params.append(param)
            
    return depth_params, pose_params


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
    # 原有参数保持不变
    parser.add_argument('--mode', type=str, choices=['generate_clean', 'adapt'], required=True, help='generate_clean:生成clean伪标签; adapt:TTA自适应')
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
    
    # 添加自监督TTA参数
    parser.add_argument('--use_ss_tta', action='store_true', help='使用自监督TTA而非EATA')
    parser.add_argument('--pose_weights_folder', type=str, default=None, help='位姿网络权重文件夹')
    parser.add_argument('--lambda_photo', type=float, default=1.0, help='光度损失权重')
    parser.add_argument('--lambda_smooth', type=float, default=0.1, help='平滑损失权重')
    parser.add_argument('--lambda_geo', type=float, default=0.5, help='几何一致性损失权重')
    
    args = parser.parse_args()

    # === 数据集 ===
    splits_dir = os.path.join(os.path.dirname(__file__), "splits")
    filenames = readlines(os.path.join(splits_dir, args.eval_split, "test_files.txt"))
    
    # 如果使用自监督TTA，需要相邻帧
    if args.use_ss_tta:
        from datasets.kitti_dataset import KITTIRAWDataset
        dataset = KITTIRAWDataset(
            args.data_path, filenames,
            args.height, args.width,
            [0, -1, 1], 4, is_train=False  # 加载当前帧和相邻帧
        )
    else:
        from datasets.kitti_dataset import KITTIRAWDataset
        dataset = KITTIRAWDataset(
            args.data_path, filenames,
            args.height, args.width,
            [0], 1, is_train=False
        )
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

    # === 加载深度网络 ===
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

    depth_model = SQLDepthModel(encoder, decoder)

    # === 如果使用自监督TTA，加载位姿网络 ===
    if args.use_ss_tta:
        # 位姿网络使用与深度网络相同架构的编码器但输入通道为6(两帧拼接)
        pose_encoder = deepcopy(encoder)
        # 修改第一层卷积以接受6通道输入
        if hasattr(pose_encoder, 'conv1'):
            old_weight = pose_encoder.conv1.weight.data
            pose_encoder.conv1 = nn.Conv2d(6, old_weight.size(0), 
                                          kernel_size=old_weight.size(2),
                                          stride=pose_encoder.conv1.stride,
                                          padding=pose_encoder.conv1.padding,
                                          bias=False)
            # 初始化新权重
            pose_encoder.conv1.weight.data[:, :3] = old_weight
            pose_encoder.conv1.weight.data[:, 3:] = old_weight
        
        # 位姿解码器
        pose_decoder = networks.PoseDecoder(
            pose_encoder.num_ch_enc if hasattr(pose_encoder, 'num_ch_enc') else [64, 64, 128, 256, 512],
            num_input_features=1,
            num_frames_to_predict_for=2
        )
        
        # 加载位姿网络权重（如果提供）
        if args.pose_weights_folder is not None:
            pose_encoder_path = os.path.join(args.pose_weights_folder, "pose_encoder.pth")
            pose_decoder_path = os.path.join(args.pose_weights_folder, "pose_decoder.pth")
            if os.path.exists(pose_encoder_path) and os.path.exists(pose_decoder_path):
                pose_encoder.load_state_dict(torch.load(pose_encoder_path))
                pose_decoder.load_state_dict(torch.load(pose_decoder_path))
        
        pose_encoder.cuda()
        pose_decoder.cuda()
        pose_model = PoseModel(pose_encoder, pose_decoder)
        
        # 配置模型进行自监督TTA
        depth_model, pose_model = configure_model_for_ss_tta(depth_model, pose_model)
        
        # 收集需要更新的参数
        depth_params, pose_params = collect_ss_tta_params(depth_model, pose_model)
        
        # 初始化优化器
        depth_optimizer = torch.optim.Adam(depth_params, lr=4e-5)
        pose_optimizer = torch.optim.Adam(pose_params, lr=1e-5)
        
        # 创建自监督TTA模型
        adapt_model = SelfSupervisedTTA(
            depth_model, pose_model, 
            depth_optimizer, pose_optimizer,
            steps=args.eata_steps,
            lambda_photo=args.lambda_photo,
            lambda_smooth=args.lambda_smooth,
            lambda_geo=args.lambda_geo
        )
    else:
        # 使用原始EATA
        configure_model_for_eata(depth_model)
        optimizer = torch.optim.Adam([p for p in depth_model.parameters() if p.requires_grad], lr=4e-5)
        adapt_model = EATA(depth_model, optimizer, steps=args.eata_steps)

    if args.mode == 'generate_clean':
        # === 生成clean伪标签 ===
        depth_model.eval()
        pred_disps = []
        with torch.no_grad():
            for batch in dataloader:
                images = batch[("color", 0, 0)].cuda()
                pred_disp = depth_model(images)
                pred_disp = pred_disp.cpu().numpy()
                if pred_disp.ndim == 4:
                    pred_disp = pred_disp[:, 0, :, :]
                pred_disps.append(pred_disp)
        pred_disps = np.concatenate(pred_disps, axis=0)
        np.save(args.clean_pred_path, pred_disps)
        print(f"Clean伪标签已保存到 {args.clean_pred_path}")
        return

    elif args.mode == 'adapt':
        # === TTA自适应推理 ===
        clean_pred_disps = np.load(args.clean_pred_path)
        
        pred_disps = []
        idx = 0
        
        for batch in dataloader:
            images = batch[("color", 0, 0)].cuda()
            # 取对应clean伪标签
            clean_depth = torch.from_numpy(clean_pred_disps[idx:idx+images.shape[0]]).cuda()
            idx += images.shape[0]
            
            if args.use_ss_tta:
                # 自监督TTA需要相邻帧和内参
                source_frames = [batch[("color", i, 0)].cuda() for i in [-1, 1] if ("color", i, 0) in batch]
                
                # 获取相机内参（如果有）
                if "K" in batch:
                    intrinsics = batch["K"].cuda()
                else:
                    intrinsics = None
                
                pred_disp = adapt_model(images, source_frames, intrinsics, clean_depth)
            else:
                # 原始EATA
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