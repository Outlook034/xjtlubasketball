import os
import sys
os.environ["MKL_NUM_THREADS"] = "1"  # noqa F402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa F402
os.environ["OMP_NUM_THREADS"] = "1"  # noqa F402
import cv2
import numpy as np
import skimage
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils import readlines
from options import MonodepthOptions
from layers import disp_to_depth, BackprojectDepth, Project3D, transformation_from_parameters
import datasets
import networks
# from .dynamicdepth import datasets, networks
# from layers import transformation_from_parameters, disp_to_depth, grad_computation_tools
import tqdm
# import matplotlib as mpl
# import matplotlib.cm as cm
import matplotlib.pyplot as plt
import time
import warnings

# 过滤torchvision的警告
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")


cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)

# VECTTA相关类定义
class SQLDepthModel(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, x):
        out = self.decoder(self.encoder(x))
        return out[("disp", 0)]  # 只返回主Tensor

class VECTTA(nn.Module):
    """几何等变一致测试时适应 (Viewpoint Equivariance Consistency Test-Time Adaptation)

    通过注入虚拟微小相机位姿变换，构建深度预测的几何闭环一致性约束。
    """
    def __init__(self, depth_model, update_mode='bn_decoder', optimizer_lr=1e-4, steps=5,
                 early_stop=True, early_stop_patience=3, grad_clip=1.0):
        super().__init__()
        self.depth_model = depth_model
        self.update_mode = update_mode
        self.optimizer_lr = optimizer_lr
        self.steps = steps
        self.early_stop = early_stop
        self.early_stop_patience = early_stop_patience
        self.grad_clip = grad_clip

        self.configure_params_to_update()
        self.optimizer = torch.optim.AdamW(self.params_to_update, lr=optimizer_lr)

    def configure_params_to_update(self):
        self.params_to_update = []
        for p in self.depth_model.parameters():
            p.requires_grad = False
            
        if self.update_mode == 'all':
            # 更新整个模型的所有参数
            for p in self.depth_model.parameters():
                p.requires_grad = True
                self.params_to_update.append(p)

        if self.update_mode == 'bn_only':
            for m in self.depth_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.requires_grad_(True)
                    m.track_running_stats = False
                    m.running_mean = None
                    m.running_var = None
                    for p in m.parameters():
                        p.requires_grad = True
                        self.params_to_update.append(p)

        if self.update_mode == 'bn_decoder':
            if hasattr(self.depth_model, 'decoder'):
                for p in self.depth_model.decoder.parameters():
                    p.requires_grad = True
                    self.params_to_update.append(p)

        if self.update_mode == 'last_layers':
            # 仅示例：匹配名称里包含"disp"的Conv层
            for name, m in self.depth_model.named_modules():
                if isinstance(m, nn.Conv2d) and 'disp' in name:
                    for p in m.parameters():
                        p.requires_grad = True
                        self.params_to_update.append(p)
            # 同时放开BN
            for m in self.depth_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.requires_grad_(True)
                    m.track_running_stats = False
                    m.running_mean = None
                    m.running_var = None
                    for p in m.parameters():
                        p.requires_grad = True
                        self.params_to_update.append(p)

    def sample_virtual_poses(self, batch_size, device=None, num_poses=4, xyz_range=0.1, rpy_range=0.05):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        patterns = [
            [xyz_range, 0, 0, 0, rpy_range, 0],
            [-xyz_range, 0, 0, 0, -rpy_range, 0],
            [0, 0, xyz_range, rpy_range, 0, 0],
            [0, xyz_range, 0, 0, 0, rpy_range],
        ]
        poses_list = []
        for tx, ty, tz, rx, ry, rz in patterns:
            translation = torch.zeros(batch_size, 1, 1, 3, device=device)
            translation[:, 0, 0, 0] = tx
            translation[:, 0, 0, 1] = ty
            translation[:, 0, 0, 2] = tz
            axisangle = torch.zeros(batch_size, 1, 1, 3, device=device)
            axisangle[:, 0, 0, 0] = rx
            axisangle[:, 0, 0, 1] = ry
            axisangle[:, 0, 0, 2] = rz
            poses_list.append((axisangle, translation))
        return poses_list

    def back_project_depth(self, depth, K, pose):
        batch_size, _, height, width = depth.shape
        device = depth.device
        K_3x3 = K[:, :3, :3] if K.shape[-1] == 4 else K
        backproject = BackprojectDepth(batch_size, height, width).to(device)
        project = Project3D(batch_size, height, width).to(device)
        inv_K = torch.inverse(K_3x3)
        K_proj = torch.zeros(batch_size, 3, 4, device=device)
        K_proj[:, :3, :3] = K_3x3
        axisangle, translation = pose
        T = transformation_from_parameters(axisangle[:, 0], translation[:, 0], invert=True)
        cam_points = backproject(depth, inv_K)
        pix_coords = project(cam_points, K_proj, T)
        z = cam_points[:, 2, :].reshape(batch_size, 1, height, width)
        transformed_depth = F.grid_sample(z, pix_coords, mode='bilinear', padding_mode='zeros', align_corners=True)
        valid_mask = ((pix_coords[:, :, :, 0] >= -1) & (pix_coords[:, :, :, 0] <= 1) &
                      (pix_coords[:, :, :, 1] >= -1) & (pix_coords[:, :, :, 1] <= 1)).unsqueeze(1).float()
        return transformed_depth * valid_mask

    def create_visibility_mask(self, depth, back_depth, border_margin=2):
        valid = torch.isfinite(back_depth) & (back_depth > 1e-6) & torch.isfinite(depth) & (depth > 1e-6)
        mask = valid.float()
        if border_margin > 0:
            mask[:, :, :border_margin, :] = 0
            mask[:, :, -border_margin:, :] = 0
            mask[:, :, :, :border_margin] = 0
            mask[:, :, :, -border_margin:] = 0
        return mask

    def compute_scale_invariant_depth_error(self, depth, back_depth, mask):
        eps = 1e-6
        depth_safe = torch.clamp(depth, min=eps)
        back_depth_safe = torch.clamp(back_depth, min=eps)
        if mask is None:
            mask = torch.ones_like(depth_safe)
        valid = mask > 0.5
        if valid.sum() == 0:
            return torch.zeros((), device=depth.device)
        log_diff = (torch.log(depth_safe) - torch.log(back_depth_safe))[valid]
        n = log_diff.numel()
        loss = (log_diff.pow(2).sum() / n) - (log_diff.sum() ** 2) / (n ** 2)
        return loss

    def compute_gradient_consistency_loss(self, depth, back_depth, mask):
        if mask is None:
            mask = torch.ones_like(depth)
        def gradient_x(img):
            return img[:, :, :, 1:] - img[:, :, :, :-1]
        def gradient_y(img):
            return img[:, :, 1:, :] - img[:, :, :-1, :]
        gx1, gy1 = gradient_x(depth), gradient_y(depth)
        gx2, gy2 = gradient_x(back_depth), gradient_y(back_depth)
        mask_x = mask[:, :, :, 1:] * mask[:, :, :, :-1]
        mask_y = mask[:, :, 1:, :] * mask[:, :, :-1, :]
        diff_x = torch.abs(gx1 - gx2) * mask_x
        diff_y = torch.abs(gy1 - gy2) * mask_y
        denom = (mask_x.sum() + mask_y.sum()).clamp(min=1.0)
        return (diff_x.sum() + diff_y.sum()) / denom

    def forward(self, image, K, clean_depth=None):
        device = image.device
        batch_size = image.shape[0]
        self.depth_model.train()
        best_loss = float('inf')
        patience_counter = 0
        
        # 记录VECTTA适应时间
        adapt_start_time = time.time()
        
        for step_idx in range(self.steps):
            step_start_time = time.time()
            
            # 需要梯度：深度必须可导，才能把几何一致性损失反传到模型参数
            depth = self.depth_model(image)
            if isinstance(depth, dict):
                depth = depth[("disp", 0)]
            poses = self.sample_virtual_poses(batch_size, device)
            total_loss = depth.new_tensor(0.0)
            for pose in poses:
                # 基于原视角深度，生成新视角"回投影"深度，避免重复图像渲染
                back_depth = self.back_project_depth(depth, K, pose)
                mask = self.create_visibility_mask(depth, back_depth)
                si_loss = self.compute_scale_invariant_depth_error(depth, back_depth, mask)
                grad_loss = self.compute_gradient_consistency_loss(depth, back_depth, mask)
                total_loss = total_loss + si_loss + 0.5 * grad_loss
            total_loss = total_loss / max(len(poses), 1)
            
            clean_loss = 0.0
            if clean_depth is not None:
                if clean_depth.dim() == 3:
                    clean_depth = clean_depth.unsqueeze(1)
                
                # 确保clean_depth和depth的形状匹配
                if clean_depth.shape != depth.shape:
                    # 调整clean_depth的形状以匹配depth
                    if clean_depth.shape[0] != depth.shape[0]:
                        # 如果batch size不匹配，取前N个样本
                        min_batch_size = min(clean_depth.shape[0], depth.shape[0])
                        clean_depth = clean_depth[:min_batch_size]
                        depth = depth[:min_batch_size]
                    
                    # 调整空间维度
                    if clean_depth.shape[2:] != depth.shape[2:]:
                        clean_depth = F.interpolate(clean_depth, size=depth.shape[2:], mode='bilinear', align_corners=False)
                
                # 只有当clean_depth不为空时才计算损失
                if clean_depth.numel() > 0 and depth.numel() > 0:
                    clean_loss = F.l1_loss(depth, clean_depth)
                    total_loss = total_loss + 0.5 * clean_loss
            
            self.optimizer.zero_grad()
            total_loss.backward()
            if self.grad_clip is not None and len(self.params_to_update) > 0:
                torch.nn.utils.clip_grad_norm_(self.params_to_update, self.grad_clip)
            self.optimizer.step()
            
            step_time = time.time() - step_start_time
            
            # 打印损失信息
            if step_idx == 0 or step_idx == self.steps - 1:
                print(f"VECTTA Step {step_idx+1}/{self.steps}: total_loss={total_loss.item():.6f}, clean_loss={clean_loss:.6f}, time={step_time:.4f}s")

        
        self.depth_model.eval()
        with torch.no_grad():
            final_depth = self.depth_model(image)
            if isinstance(final_depth, dict):
                final_depth = final_depth[("disp", 0)]
        return final_depth

class EATA(nn.Module):
    def __init__(self, model, optimizer, steps=1):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.criterion = nn.L1Loss()
        
    def forward(self, x, clean_depth=None):
        self.model.train()
        
        # 记录EATA适应时间
        adapt_start_time = time.time()
        
        for step_idx in range(self.steps):
            step_start_time = time.time()
            
            out = self.model(x)
            if clean_depth is not None:
                if clean_depth.dim() == 3:
                    clean_depth = clean_depth.unsqueeze(1)  # [B, 1, H, W]
                # 假设out和clean_depth shape一致
                loss = self.criterion(out, clean_depth)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                
            step_time = time.time() - step_start_time
            if step_idx == 0:
                print(f"EATA Step {step_idx+1}/{self.steps} time: {step_time:.4f}s")
        
        adapt_time = time.time() - adapt_start_time
        print(f"EATA total adaptation time: {adapt_time:.4f}s")
        
        self.model.eval()
        with torch.no_grad():
            return self.model(x)

def configure_model_for_eata(model):
    for param in model.parameters():
        param.requires_grad = False
    if hasattr(model, 'decoder'):
        for p in model.decoder.parameters():
            p.requires_grad = True
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
            for p in m.parameters():
                p.requires_grad = True
    return model

splits_dir = "splits"

# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4

_DEPTH_COLORMAP = plt.get_cmap('plasma', 256)
def colormap(inputs, normalize=True, torch_transpose=True):
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.detach().cpu().numpy()

    vis = inputs
    if normalize:
        ma = float(vis.max())
        mi = float(vis.min())
        d = ma - mi if ma != mi else 1e5
        vis = (vis - mi) / d

    if vis.ndim == 4:
        vis = vis.transpose([0, 2, 3, 1])
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[:, :, :, 0, :3]
        if torch_transpose:
            vis = vis.transpose(0, 3, 1, 2)
    elif vis.ndim == 3:
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[:, :, :, :3]
        if torch_transpose:
            vis = vis.transpose(0, 3, 1, 2)
    elif vis.ndim == 2:
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[..., :3]
        if torch_transpose:
            vis = vis.transpose(2, 0, 1)

    return vis


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
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


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

def compute_matching_mask(monodepth, lowest_cost):
        """Generate a mask of where we cannot trust the cost volume, based on the difference
        between the cost volume and the teacher, monocular network"""

        matching_depth = 1 / lowest_cost.unsqueeze(1).cuda()

        # mask where they differ by a large amount
        mask = ((matching_depth - monodepth) / monodepth) < 1.0
        mask *= ((monodepth - matching_depth) / matching_depth) < 1.0
        return mask[:, 0]


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    frames_to_load = [0]
    # if opt.use_future_frame == 'true':
    #     frames_to_load.append(1)
    # for idx in range(-1, -1 - opt.num_matching_frames, -1):
    #     if idx not in frames_to_load:
    #         frames_to_load.append(idx)

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"
    
    # VECTTA相关参数
    use_vectta = getattr(opt, 'use_vectta', False)
    vectta_mode = getattr(opt, 'vectta_mode', 'vec')  # 'vec' or 'eata'
    clean_pred_path = getattr(opt, 'clean_pred_path', 'clean_pred_disps.npy')
    generate_clean = getattr(opt, 'generate_clean', False)
    mode = getattr(opt, 'mode', None)
    
    # 根据mode参数设置generate_clean
    if mode == 'generate_clean':
        generate_clean = True
    elif mode == 'adapt':
        use_vectta = True
    
    # VECTTA参数
    vec_steps = getattr(opt, 'vec_steps', 5)
    vec_update_mode = getattr(opt, 'vec_update_mode', 'bn_decoder')
    vec_lr = getattr(opt, 'vec_lr', 1e-4)
    vec_early_stop = getattr(opt, 'vec_early_stop', True)
    vec_early_stop_patience = getattr(opt, 'vec_early_stop_patience', 3)
    vec_grad_clip = getattr(opt, 'vec_grad_clip', 1.0)
    
    # EATA参数
    eata_steps = getattr(opt, 'eata_steps', 1)

    if opt.ext_disp_to_eval is None:

        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)

        print("-> Loading weights from {}".format(opt.load_weights_folder))

        # Setup dataloaders
        filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

        encoder_dict = torch.load(encoder_path)

        try:
            HEIGHT, WIDTH = encoder_dict['height'], encoder_dict['width']
        except KeyError:
            print('No "height" or "width" keys found in the encoder state_dict, resorting to '
                  'using command line values!')
            HEIGHT, WIDTH = opt.height, opt.width
        
        if opt.eval_split == 'cityscapes':
            dataset = datasets.CityscapesEvalDataset(opt.eval_data_path, filenames,
                                                     HEIGHT, WIDTH,
                                                     frames_to_load, 4,
                                                     is_train=False)
        else:
            dataset = datasets.KITTIRAWDataset(opt.data_path, filenames,
                                               encoder_dict['height'], encoder_dict['width'],
                                               frames_to_load, 4,
                                               is_train=False)
        dataloader = DataLoader(dataset, 4, shuffle=False, num_workers=8,
                                pin_memory=True, drop_last=False)


        # encoder = networks.BaseEncoder.build(model_dim=opt.model_dim)
        encoder = networks.Resnet50EncoderDecoder(model_dim=opt.model_dim)
        depth_decoder = networks.Depth_Decoder_QueryTr(in_channels=opt.model_dim, patch_size=opt.patch_size, dim_out=opt.dim_out, embedding_dim=opt.model_dim, 
                                                                query_nums=opt.query_nums, num_heads=4,
                                                                min_val=opt.min_depth, max_val=opt.max_depth)
                                                                # min_val=0.001, max_val=10.0)

        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict(torch.load(decoder_path))

        encoder.cuda()
        encoder = torch.nn.DataParallel(encoder)
        encoder.eval()
        depth_decoder.cuda()
        depth_decoder = torch.nn.DataParallel(depth_decoder)
        depth_decoder.eval()

        # 创建SQLDepthModel用于VECTTA
        if use_vectta:
            # 创建不包装DataParallel的模型用于VECTTA
            encoder_single = networks.Resnet50EncoderDecoder(model_dim=opt.model_dim)
            encoder_single.load_state_dict({k: v for k, v in encoder_dict.items() if k in encoder_single.state_dict()})
            encoder_single.cuda()
            encoder_single.eval()
            
            depth_decoder_single = networks.Depth_Decoder_QueryTr(in_channels=opt.model_dim, patch_size=opt.patch_size, dim_out=opt.dim_out, embedding_dim=opt.model_dim, 
                                                                 query_nums=opt.query_nums, num_heads=4,
                                                                 min_val=opt.min_depth, max_val=opt.max_depth)
            depth_decoder_single.load_state_dict(torch.load(decoder_path))
            depth_decoder_single.cuda()
            depth_decoder_single.eval()
            
            # 创建SQLDepthModel
            depth_model = SQLDepthModel(encoder_single, depth_decoder_single)
            
            # 根据模式配置VECTTA或EATA
            if vectta_mode == 'vec':
                adapt_model = VECTTA(
                    depth_model=depth_model,
                    update_mode=vec_update_mode,
                    optimizer_lr=vec_lr,
                    steps=vec_steps,
                    early_stop=vec_early_stop,
                    early_stop_patience=vec_early_stop_patience,
                    grad_clip=vec_grad_clip,
                )
                adapt_model.cuda()
            else:  # EATA模式
                configure_model_for_eata(depth_model)
                optimizer = torch.optim.Adam([p for p in depth_model.parameters() if p.requires_grad], lr=4e-5)
                adapt_model = EATA(depth_model, optimizer, steps=eata_steps)
            
            # 加载clean伪标签
            if os.path.exists(clean_pred_path):
                clean_pred_disps = np.load(clean_pred_path)
                print(f"Loaded clean predictions from {clean_pred_path}")
            else:
                print(f"Warning: Clean predictions file {clean_pred_path} not found. VECTTA will run without clean supervision.")
                clean_pred_disps = None

        # if torch.cuda.is_available():
        #     encoder.cuda()
        #     depth_decoder.cuda()

        pred_disps = []
        src_imgs = []
        error_maps = []
        # mono_disps = []
        
        # 初始化VECTTA相关变量（如果还没有初始化）
        if not use_vectta:
            clean_pred_disps = None
            adapt_model = None

        if opt.eval_split == 'cityscapes':
            print('loading cityscapes gt depths individually due to their combined size!')
            gt_depths = os.path.join(splits_dir, opt.eval_split, "gt_depths")
        else:
            gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
            gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

        print("-> Computing predictions with size {}x{}".format(HEIGHT, WIDTH))
        
        # 生成clean伪标签模式
        if generate_clean:
            print("-> Generating clean predictions...")
            encoder.eval()
            depth_decoder.eval()
            pred_disps = []
            with torch.no_grad():
                for i, data in tqdm.tqdm(enumerate(dataloader)):
                    input_color = data[('color', 0, 0)]
                    if torch.cuda.is_available():
                        input_color = input_color.cuda()
                    
                    if opt.post_process:
                        # Post-processed results require each image to have two forward passes
                        input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

                    output = depth_decoder(encoder(input_color))
                    pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
                    pred_disp = pred_disp.cpu()[:, 0].numpy()

                    if opt.post_process:
                        n = pred_disp.shape[0] // 2
                        pred_disp = batch_post_process_disparity(pred_disp[:n], pred_disp[n:, :, ::-1])

                    pred_disps.append(pred_disp)
            
            pred_disps = np.concatenate(pred_disps)
            np.save(clean_pred_path, pred_disps)
            print(f"-> Clean predictions saved to {clean_pred_path}")
            print(f"-> Generated {pred_disps.shape[0]} clean predictions")
            return
        
        if use_vectta:
            print(f"-> Using VECTTA mode: {vectta_mode}")

        # do inference
        if use_vectta:
            # 使用VECTTA进行推理
            clean_idx = 0
            for i, data in tqdm.tqdm(enumerate(dataloader)):
                input_color = data[('color', 0, 0)]
                K = data[('K', 0)]
                if torch.cuda.is_available():
                    input_color = input_color.cuda()
                    K = K.cuda()
                
                if opt.post_process:
                    # Post-processed results require each image to have two forward passes
                    input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)
                    K = torch.cat((K, K), 0)  # 复制K矩阵

                # 准备clean_depth
                clean_depth = None
                if clean_pred_disps is not None:
                    batch_size = input_color.shape[0]
                    if clean_idx + batch_size <= len(clean_pred_disps):
                        clean_depth = torch.from_numpy(clean_pred_disps[clean_idx:clean_idx+batch_size]).cuda()
                        clean_idx += batch_size
                    else:
                        # 如果clean数据不够，取最后一部分
                        remaining = len(clean_pred_disps) - clean_idx
                        if remaining > 0:
                            clean_depth = torch.from_numpy(clean_pred_disps[clean_idx:]).cuda()
                            clean_depth = torch.cat([clean_depth, clean_depth[-1:].repeat(batch_size - remaining, 1, 1)], dim=0)
                        clean_idx = len(clean_pred_disps)

                # 使用VECTTA进行推理
                if vectta_mode == 'vec':
                    pred_disp = adapt_model(input_color, K, clean_depth=clean_depth)
                else:  # EATA模式
                    pred_disp = adapt_model(input_color, clean_depth=clean_depth)

                pred_disp = pred_disp.cpu()[:, 0].numpy()

                if opt.post_process:
                    n = pred_disp.shape[0] // 2
                    pred_disp = batch_post_process_disparity(pred_disp[:n], pred_disp[n:, :, ::-1])

                pred_disps.append(pred_disp)
        else:
            # 原始推理方式
            with torch.no_grad():
                for i, data in tqdm.tqdm(enumerate(dataloader)):
                    input_color = data[('color', 0, 0)]
                    if torch.cuda.is_available():
                        input_color = input_color.cuda()
                        # print(input_color.shape, "==") # [16, 3, 192, 512]

                    
                    if opt.post_process:
                        # print("post_process *********")
                        # Post-processed results require each image to have two forward passes
                        input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

                    output = depth_decoder(encoder(input_color))

                    pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
                    # pred_disp = output[("disp", 0)]
                    pred_disp = pred_disp.cpu()[:, 0].numpy()

                    if opt.post_process:
                        n = pred_disp.shape[0] // 2
                        pred_disp = batch_post_process_disparity(pred_disp[:n], pred_disp[n:, :, ::-1])

                    pred_disps.append(pred_disp)
                    # src_imgs.append(data[("color", 0, 0)])

                    # monodisp, _ = disp_to_depth(teacher_output[("disp", 0)], opt.min_depth, opt.max_depth)
                    # monodisp = monodisp.cpu()[:, 0].numpy()
                    # mono_disps.append(monodisp)
                    

        pred_disps = np.concatenate(pred_disps)
        # src_imgs = np.concatenate(src_imgs)

        print('finished predicting!')
        if opt.save_pred_disps:
            output_path = os.path.join(
                opt.load_weights_folder, "disps_{}_split.npy".format(opt.eval_split))
            print("-> Saving predicted disparities to ", output_path)
            np.save(output_path, pred_disps)

            # src_imgs_path = os.path.join(
            #     opt.load_weights_folder, "src_{}_split.npy".format(opt.eval_split))
            # print("-> Saving src imgs to ", src_imgs_path)
            # np.save(src_imgs_path, src_imgs)

    else:
        # Load predictions from file
        print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
        pred_disps = np.load(opt.ext_disp_to_eval)

        if opt.eval_eigen_to_benchmark:
            eigen_to_benchmark_ids = np.load(
                os.path.join(splits_dir, "benchmark", "eigen_to_benchmark_ids.npy"))

            pred_disps = pred_disps[eigen_to_benchmark_ids]


    print("-> Evaluating")

    if opt.eval_stereo:
        print("   Stereo evaluation - "
              "disabling median scaling, scaling by {}".format(STEREO_SCALE_FACTOR))
        opt.disable_median_scaling = True
        opt.pred_depth_scale_factor = STEREO_SCALE_FACTOR
    else:
        print("   Mono evaluation - using median scaling")

    errors = []
    ratios = []
    for i in tqdm.tqdm(range(pred_disps.shape[0])):

        if opt.eval_split == 'cityscapes':
            gt_depth = np.load(os.path.join(gt_depths, str(i).zfill(3) + '_depth.npy'))
            gt_height, gt_width = gt_depth.shape[:2]
            # print(gt_height, "print(gt_height)") 1024
            # print(gt_width, "print(gt_width)") 2048
            # crop ground truth to remove ego car -> this has happened in the dataloader for input
            # images
            gt_height = int(round(gt_height * 0.75))
            gt_depth = gt_depth[:gt_height]

        else:
            gt_depth = gt_depths[i]
            gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = np.squeeze(pred_disps[i])
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = pred_disp
        # pred_depth = 1 / pred_disp

        # mono_disp = np.squeeze(mono_disps[i][0])
        # mono_disp = cv2.resize(mono_disp, (gt_width, gt_height))
        # mono_depth = 1 / mono_disp

        if opt.eval_split == 'cityscapes':
            # when evaluating cityscapes, we centre crop to the middle 50% of the image.
            # Bottom 25% has already been removed - so crop the sides and the top here
            gt_depth = gt_depth[256:, 192:1856]
            pred_depth = pred_depth[256:, 192:1856]
            # 768, 2048
            # mono_depth = mono_depth[256:, 192:1856]

        if opt.eval_split == "eigen":
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

        elif opt.eval_split == 'cityscapes':
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

        else:
            mask = gt_depth > 0
        
        if not opt.disable_median_scaling:
            ratio = np.median(gt_depth[mask]) / np.median(pred_depth[mask])
            ratios.append(ratio)
            pred_depth *= ratio

            # mono_depth *= np.median(gt_depth[mask]) / np.median(mono_depth[mask])
        
        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        # mono_depth[mono_depth < MIN_DEPTH] = MIN_DEPTH
        # mono_depth[mono_depth > MAX_DEPTH] = MAX_DEPTH
        
        error_map = np.abs(gt_depth - pred_depth)
        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]
        error_map = np.multiply(error_map, mask)
        error_maps.append(error_map)

        errors.append(compute_errors(gt_depth, pred_depth))

    if opt.save_pred_disps:
        print("saving errors")
        # if opt.zero_cost_volume:
        if True:
            tag = "mono"
        else:
            tag = "multi"
        output_path = os.path.join(
            opt.load_weights_folder, "{}_{}_errors.npy".format(tag, opt.eval_split))
        np.save(output_path, np.array(errors))

    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    if opt.save_pred_disps:
        # error_maps = np.concatenate(error_maps) # should not concatenate
        error_map_path = os.path.join(
            opt.load_weights_folder, "error_{}_split".format(opt.eval_split))
        print("-> Saving error maps to ", error_map_path)
        np.save(error_map_path, error_maps)
        # np.savez_compressed(error_map_path, data=np.array(error_maps, dtype="object")) # do not use
    mean_errors = np.array(errors).mean(0)
    print(mean_errors)

    print("\n  " + ("{:>8} | " * 7).format("abs_rel",
                                           "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")


def convert_arg_line_to_args(arg_line):
    args = []
    for arg in arg_line.split():
        if not arg.strip():
            continue
        args.append(str(arg))
    return args

if __name__ == "__main__":
    options = MonodepthOptions()
    options.parser.convert_arg_line_to_args = convert_arg_line_to_args
    if sys.argv.__len__() == 2:
        arg_filename_with_prefix = '@' + sys.argv[1]
        opt = options.parser.parse_args([arg_filename_with_prefix])
    else:
        opt = options.parser.parse_args()
    evaluate(opt)
    # options = MonodepthOptions()
    # evaluate(options.parse())


