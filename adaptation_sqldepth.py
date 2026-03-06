import os
import copy
import time
import math
import numpy as np
import cv2
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

import datasets
import networks
from layers import (
    disp_to_depth, BackprojectDepth, Project3D, transformation_from_parameters,
    get_smooth_loss, compute_depth_errors, SSIM
)
from utils import readlines
from SQLdepth import MonodepthOptions

# 导入 DepthDecoder（用于正则模型）
import sys
import os

# 从 SfMNeXt-Impl 到 ada-depth-main/ada-depth-main 的路径
script_dir = os.path.dirname(os.path.abspath(__file__))  # SfMNeXt-Impl 目录
parent_dir = os.path.dirname(script_dir)  # ada-depth-main 根目录
ada_depth_path = os.path.join(parent_dir, 'ada-depth-main')  # ../ada-depth-main
ada_depth_path = os.path.abspath(ada_depth_path)

# 移除已导入的 SfMNeXt-Impl 的 networks 模块（第14行导入的），
# 这样才能导入 ada-depth-main 的 networks 模块
if 'networks' in sys.modules:
    if hasattr(sys.modules['networks'], '__file__'):
        if 'SfMNeXt-Impl' in str(sys.modules['networks'].__file__):
            del sys.modules['networks']  # 删除 networks 模块
            for k in list(sys.modules.keys()):
                if k.startswith('networks.'):  # 删除 networks 的所有子模块
                    del sys.modules[k]

# 将 ada-depth-main 添加到 sys.path 最前面
if ada_depth_path not in sys.path:
    sys.path.insert(0, ada_depth_path)
elif sys.path.index(ada_depth_path) != 0:
    sys.path.remove(ada_depth_path)
    sys.path.insert(0, ada_depth_path)

# 直接导入
try:
    # 修复 torchvision 兼容性问题
    import torchvision.models as models
    import torch.utils.model_zoo as model_zoo
    # 如果 model_urls 不存在，创建一个兼容的字典
    if not hasattr(models.resnet, 'model_urls'):
        models.resnet.model_urls = {
            'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
            'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
            'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
            'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
            'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
        }
    
    from networks.depth_decoder import DepthDecoder
    from networks.resnet_encoder import ResnetEncoder
    from networks.pose_decoder import PoseDecoder as AdaPoseDecoder
except ImportError:
    DepthDecoder = None
    ResnetEncoder = None
    AdaPoseDecoder = None
    print(f"Failed to import from ada-depth-main. Path: {ada_depth_path}")


def update_ema_variables(model, ema_model, alpha, global_step):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha)
        ema_param.data.add_((1 - alpha) * param.data)


class AdaptSQLdepth:
    """将 ada-depth 的适应流程迁移到 SQLdepth 模型"""
    def __init__(self, options):
        self.opt = options
        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")
        self.MIN_DEPTH = 1e-3
        self.MAX_DEPTH = 80

        self.h_KITTI = 1.65
        self.input_height, self.input_width = None, None
        if self.opt.dataset == "kitti":
            self.input_height, self.input_width = 352, 1216
            self.ratio = 1
            self.cam_h = self.h_KITTI

        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        self.depth_metric_names = ["sup/abs_rel", "sup/sq_rel", "sup/rms", "sup/log_rms",
                                   "sup/a1", "sup/a2", "sup/a3", "median/global"]
        self.depth_metric_names_local = ["supl/abs_rel", "supl/sq_rel", "supl/rms", "supl/log_rms",
                                         "supl/a1", "supl/a2", "supl/a3", "median/local"]
        self.depth_metric_names_unsup = ["unsupl/abs_rel", "unsupl/sq_rel", "unsupl/rms",
                                          "unsupl/log_rms", "unsupl/a1", "unsupl/a2", "unsupl/a3",
                                          "median/unsup"]

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1).to(self.device)

        self.ssim = SSIM()
        self.ssim.to(self.device)

        # === 主模型：使用 SQLdepth 架构 ===
        self.models = {}
        self.parameters_to_train = []
        
        # SQLdepth encoder
        if self.opt.backbone in ["resnet", "resnet_lite"]:
            self.models["encoder"] = networks.ResnetEncoderDecoder(
                num_layers=self.opt.num_layers,
                num_features=self.opt.num_features,
                model_dim=self.opt.model_dim
            )
        elif self.opt.backbone == "resnet18_lite":
            self.models["encoder"] = networks.LiteResnetEncoderDecoder(model_dim=self.opt.model_dim)
        elif self.opt.backbone == "eff_b5":
            self.models["encoder"] = networks.BaseEncoder.build(
                num_features=self.opt.num_features, 
                model_dim=self.opt.model_dim
            )
        else:
            self.models["encoder"] = networks.Unet(
                pretrained=(not self.opt.load_pretrained_model),
                backbone=self.opt.backbone,
                in_channels=3,
                num_classes=self.opt.model_dim,
                decoder_channels=self.opt.dec_channels
            )
        
        # SQLdepth decoder
        if self.opt.backbone.endswith("_lite"):
            self.models["depth"] = networks.Lite_Depth_Decoder_QueryTr(
                in_channels=self.opt.model_dim,
                patch_size=self.opt.patch_size,
                dim_out=self.opt.dim_out,
                embedding_dim=self.opt.model_dim,
                query_nums=self.opt.query_nums,
                num_heads=4,
                min_val=self.opt.min_depth,
                max_val=self.opt.max_depth
            )
        else:
            self.models["depth"] = networks.Depth_Decoder_QueryTr(
                in_channels=self.opt.model_dim,
                patch_size=self.opt.patch_size,
                dim_out=self.opt.dim_out,
                embedding_dim=self.opt.model_dim,
                query_nums=self.opt.query_nums,
                num_heads=4,
                min_val=self.opt.min_depth,
                max_val=self.opt.max_depth
            )
        
        self.models["encoder"] = torch.nn.DataParallel(self.models["encoder"])
        self.models["depth"] = torch.nn.DataParallel(self.models["depth"])
        self.models["encoder"].to(self.device)
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())
        self.parameters_to_train += list(self.models["depth"].parameters())

        load_model_list = ['encoder', 'depth']
        if self.opt.load_weights_folder is not None:
            self.load_model(self.opt.load_weights_folder, self.models, load_model_list)

        # EMA models for regularisation of global scale
        self.models_ema = copy.deepcopy(self.models)
        for m in self.models_ema.values():
            m.eval()

        # Reference models for improvements calculation
        self.models_ref = copy.deepcopy(self.models)
        self.models_ref_state = {}
        for key, value in self.models_ref.items():
            self.models_ref_state[key] = copy.deepcopy(self.models[key].state_dict())
            value.eval()

        # === 正则模型：用于自监督（可以使用 SQLdepth 或保持 ResNet） ===
        print("loading regularisation model from " + self.opt.reg_path)
        self.reg_models = {}
        self.reg_parameters_to_train = []
        
        # 正则模型使用 ResNet + DepthDecoder（保持与 ada-depth 一致）
        if DepthDecoder is None:
            raise ImportError("DepthDecoder is required for regularization models. Please ensure ada-depth-main is available.")
        
        self.reg_models["encoder"] = ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained")
        self.reg_models["encoder"] = torch.nn.DataParallel(self.reg_models["encoder"])
        self.reg_models["encoder"].to(self.device)
        self.reg_models["depth"] = DepthDecoder(
            self.reg_models["encoder"].module.num_ch_enc, self.opt.scales)
        self.reg_models["depth"] = torch.nn.DataParallel(self.reg_models["depth"])
        self.reg_models["depth"].to(self.device)
        self.reg_models["pose_encoder"] = ResnetEncoder(
            self.opt.num_layers,
            self.opt.weights_init == "pretrained",
            num_input_images=self.num_pose_frames)
        self.reg_models["pose_encoder"] = torch.nn.DataParallel(self.reg_models["pose_encoder"])
        self.reg_models["pose_encoder"].to(self.device)
        self.reg_models["pose"] = AdaPoseDecoder(
            self.reg_models["pose_encoder"].module.num_ch_enc,
            num_input_features=1,
            num_frames_to_predict_for=2)
        self.reg_models["pose"] = torch.nn.DataParallel(self.reg_models["pose"])
        self.reg_models["pose"].to(self.device)

        self.reg_parameters_to_train += list(self.reg_models["encoder"].parameters())
        self.reg_parameters_to_train += list(self.reg_models["depth"].parameters())
        self.reg_parameters_to_train += list(self.reg_models["pose_encoder"].parameters())
        self.reg_parameters_to_train += list(self.reg_models["pose"].parameters())

        reg_model_folder = os.path.join(self.opt.reg_path)
        self.load_model(reg_model_folder, self.reg_models, ['encoder', 'depth', 'pose_encoder', 'pose'])

        self.reg_models_ref = copy.deepcopy(self.reg_models)
        for m in self.reg_models_ref.values():
            m.eval()

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.reg_model_optimizer = optim.Adam(self.reg_parameters_to_train, self.opt.learning_rate)

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # Dataloader
        splits_dir = './splits'
        filenames = readlines(os.path.join(splits_dir, self.opt.eval_split, "test_files.txt"))
        # 过滤掉 frame_index = 0 的样本，避免访问负数索引（仿照 VECTTA 的处理逻辑）
        filtered_filenames = []
        for fname in filenames:
            parts = fname.split()
            if len(parts) >= 2:
                frame_idx = int(parts[1])
                if frame_idx > 0:  # 只保留 frame_index > 0 的样本
                    filtered_filenames.append(fname)
            else:
                filtered_filenames.append(fname)
        self.filenames = filtered_filenames
        print(f"Loaded {len(self.filenames)} samples (filtered from {len(filenames)} total)")

    def run_adapt(self):
        self.start_time = time.time()
        self.step = 0

        errors_tt, errors_local_tt, errors_ref_tt = [], [], []
        errors_t, errors_local_t, errors_unsup_t = [], [], []
        errors_ref_t, errors_local_ref_t, errors_unsup_ref_t, errors_teacher_t, errors_teacher_local_t = [], [], [], [], []

        # 使用 SfMNeXt-Impl 的 KITTI 数据集（只使用当前帧，仿照 VECTTA）
        dataset = datasets.KITTIRAWDataset(
            self.opt.data_path, self.filenames,
            self.input_height, self.input_width,
            self.opt.frame_ids, self.num_scales,
            is_train=True
        )
        
        self.dataloader = DataLoader(dataset, 1, shuffle=False,
                                     num_workers=self.opt.num_workers,
                                     pin_memory=True, drop_last=False)

        pbar = tqdm(total=len(self.dataloader))
        errors, errors_local, errors_unsup = [], [], []
        errors_ref, errors_local_ref, errors_unsup_ref, errors_teacher, errors_teacher_local = [], [], [], [], []

        for batch_idx, inputs in enumerate(self.dataloader):
            # 适配数据格式：SfMNeXt-Impl 数据集返回 color，需要转换为 color_uncrop
            if "color_uncrop" not in inputs:
                # 将 color 映射到 color_uncrop
                for key in list(inputs.keys()):
                    if isinstance(key, tuple) and key[0] == "color":
                        inputs[("color_uncrop", key[1], key[2])] = inputs[key]
            
            # 添加 depth_gt_uncrop（如果存在 depth_gt）
            if "depth_gt" in inputs and "depth_gt_uncrop" not in inputs:
                depth_gt = inputs["depth_gt"]
                if depth_gt.dim() == 3:
                    depth_gt = depth_gt.unsqueeze(0)
                inputs["depth_gt_uncrop"] = depth_gt
            
            # 添加 height_sec 和 width_sec
            if "height_sec" not in inputs:
                inputs["height_sec"] = torch.tensor([self.input_height])
            if "width_sec" not in inputs:
                inputs["width_sec"] = torch.tensor([self.input_width])
            
            self.height = inputs["height_sec"].to(self.device)
            self.width = inputs["width_sec"].to(self.device)

            # Checking height and width are multiples of 32
            assert self.height % 32 == 0, "'height' must be a multiple of 32"
            assert self.width % 32 == 0, "'width' must be a multiple of 32"

            self.backproject_depth = {}
            self.project_3d = {}
            for scale in self.opt.scales:
                h = self.height // (2 ** scale)
                w = self.width // (2 ** scale)

                self.backproject_depth[scale] = BackprojectDepth(1, h, w)
                self.backproject_depth[scale].to(self.device)

                self.project_3d[scale] = Project3D(1, h, w)
                self.project_3d[scale].to(self.device)

            self.step += 1

            error, error_local, error_unsup, outputs, losses,\
            error_ref, error_local_ref, error_unsup_ref, error_teacher,\
            error_teacher_local = self.process_batch(inputs)

            errors.append(error)
            errors_teacher.append(error_teacher)
            errors_local.append(error_local)
            errors_teacher_local.append(error_teacher_local)
            errors_unsup.append(error_unsup)
            errors_ref.append(error_ref)
            errors_local_ref.append(error_local_ref)
            errors_unsup_ref.append(error_unsup_ref)

            errors_tt.append(error)
            errors_local_tt.append(error_local)
            errors_ref_tt.append(error_ref)

            if batch_idx % 100 == 0:
                mean_errors_100 = np.array(errors_tt).mean(0)
                mean_errors_local_100 = np.array(errors_local_tt).mean(0)
                mean_errors_unsup_100 = np.array(errors_ref_tt).mean(0)

            pbar.update(1)
            pbar.set_description("abs_rel diff: {:.4f}, abs_rel ref: {:.4f}".format(
                error[0]-error_ref[0], error_ref[0]))

        mean_errors = np.array(errors).mean(0)
        mean_errors_teacher = np.array(errors_teacher).mean(0)
        mean_errors_local = np.array(errors_local).mean(0)
        mean_errors_teacher_local = np.array(errors_teacher_local).mean(0)
        mean_errors_unsup = np.array(errors_unsup).mean(0)
        mean_errors_ref = np.array(errors_ref).mean(0)
        mean_errors_local_ref = np.array(errors_local_ref).mean(0)
        mean_errors_unsup_ref = np.array(errors_unsup_ref).mean(0)
        pbar.close()

        print("\n  " + ("{:>8} | " * 8).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3", "median"))
        print("supervised w/o gt median scaling (inital/teacher/student)")
        print(("&{: 8.3f}  " * 8).format(*mean_errors_ref.tolist()) + "\\\\")
        print(("&{: 8.3f}  " * 8).format(*mean_errors_teacher.tolist()) + "\\\\")
        print(("&{: 8.3f}  " * 8).format(*mean_errors.tolist()) + "\\\\")
        print("supervised w gt median scaling (inital/teacher/student)")
        print(("&{: 8.3f}  " * 8).format(*mean_errors_local_ref.tolist()) + "\\\\")
        print(("&{: 8.3f}  " * 8).format(*mean_errors_teacher_local.tolist()) + "\\\\")
        print(("&{: 8.3f}  " * 8).format(*mean_errors_local.tolist()) + "\\\\")
        print("self-supervised w gt median scaling (inital/adapted)")
        print(("&{: 8.3f}  " * 8).format(*mean_errors_unsup_ref.tolist()) + "\\\\")
        print(("&{: 8.3f}  " * 8).format(*mean_errors_unsup.tolist()) + "\\\\")
        print("\n-> Done!")

    def process_batch(self, inputs):
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        # First update the self-supervised models
        for m in self.reg_models.values():
            m.train()

        reg_features = self.reg_models["encoder"]((inputs["color_uncrop", 0, 0]-self.mean)/self.std)
        reg_outputs = self.reg_models["depth"](reg_features)
        reg_outputs.update(self.predict_poses(inputs, reg_features, self.reg_models))
        _, reg_depth_unsup = disp_to_depth(reg_outputs[("disp", 0)], self.opt.min_depth, self.opt.max_depth)
        self.generate_images_pred(inputs, reg_outputs)
        unsup_losses = self.compute_losses_unsup(inputs, reg_outputs)
        self.reg_model_optimizer.zero_grad()
        unsup_losses["loss"].backward()
        self.reg_model_optimizer.step()

        for m in self.reg_models.values():
            m.eval()

        for m in self.models.values():
            m.train()

        # Run adaptation with pseudo labels
        # SQLdepth 模型前向传播
        features = self.models["encoder"]((inputs["color_uncrop", 0, 0]-self.mean)/self.std)
        depth_output = self.models["depth"](features)
        # SQLdepth decoder 返回字典，需要提取 disp
        if isinstance(depth_output, dict):
            depth = depth_output[("disp", 0)]
        else:
            depth = depth_output
        
        with torch.no_grad():
            reg_features = self.reg_models["encoder"]((inputs["color_uncrop", 0, 0]-self.mean)/self.std)
            reg_outputs = self.reg_models["depth"](reg_features)
            reg_outputs.update(self.predict_poses(inputs, reg_features, self.reg_models))
            _, reg_depth_unsup = disp_to_depth(reg_outputs[("disp", 0)], self.opt.min_depth, self.opt.max_depth)

            # Reference models
            features_uncrop_ref = self.models_ref["encoder"]((inputs["color_uncrop", 0, 0]-self.mean)/self.std)
            depth_uncrop_ref_output = self.models_ref["depth"](features_uncrop_ref)
            if isinstance(depth_uncrop_ref_output, dict):
                depth_uncrop_ref = depth_uncrop_ref_output[("disp", 0)]
            else:
                depth_uncrop_ref = depth_uncrop_ref_output
            _, depth_uncrop_ref = disp_to_depth(depth_uncrop_ref, self.opt.min_depth, self.opt.max_depth)

            features_uncrop_ema = self.models_ema["encoder"]((inputs["color_uncrop", 0, 0]-self.mean)/self.std)
            depth_uncrop_ema_output = self.models_ema["depth"](features_uncrop_ema)
            if isinstance(depth_uncrop_ema_output, dict):
                depth_uncrop_ema = depth_uncrop_ema_output[("disp", 0)]
            else:
                depth_uncrop_ema = depth_uncrop_ema_output
            _, depth_uncrop_ema = disp_to_depth(depth_uncrop_ema, self.opt.min_depth, self.opt.max_depth)

            reg_features_ref = self.reg_models_ref["encoder"]((inputs["color_uncrop", 0, 0]-self.mean)/self.std)
            reg_outputs_ref = self.reg_models_ref["depth"](reg_features_ref)
            _, depth_unsup_ref = disp_to_depth(reg_outputs_ref[("disp", 0)], self.opt.min_depth, self.opt.max_depth)

            pseudo_depth_sup, pseudo_depth_unsup = self.augment_pseudo(inputs,
                                                                      depth_uncrop_ref,
                                                                      reg_depth_unsup,
                                                                      depth_uncrop_ema)

        # Convert depth (disparity) to depth for loss computation
        _, depth_pred = disp_to_depth(depth, self.opt.min_depth, self.opt.max_depth)
        
        loss_ada = self.compute_losses(depth_pred, pseudo_depth_sup, pseudo_depth_unsup)
        self.model_optimizer.zero_grad()
        loss_ada.backward()
        self.model_optimizer.step()

        # Update EMA models
        update_ema_variables(self.models["encoder"],
                             self.models_ema["encoder"],
                             0.99,
                             self.step)
        update_ema_variables(self.models["depth"],
                             self.models_ema["depth"],
                             0.99,
                             self.step)

        for m in self.models.values():
            m.eval()

        with torch.no_grad():
            features_uncrop = self.models["encoder"]((inputs["color_uncrop", 0, 0]-self.mean)/self.std)
            depth_uncrop_output = self.models["depth"](features_uncrop)
            if isinstance(depth_uncrop_output, dict):
                depth_uncrop = depth_uncrop_output[("disp", 0)]
            else:
                depth_uncrop = depth_uncrop_output
            _, depth_uncrop = disp_to_depth(depth_uncrop, self.opt.min_depth, self.opt.max_depth)

            features_uncrop_ema = self.models_ema["encoder"]((inputs["color_uncrop", 0, 0]-self.mean)/self.std)
            depth_uncrop_ema_output = self.models_ema["depth"](features_uncrop_ema)
            if isinstance(depth_uncrop_ema_output, dict):
                depth_uncrop_ema = depth_uncrop_ema_output[("disp", 0)]
            else:
                depth_uncrop_ema = depth_uncrop_ema_output
            _, depth_uncrop_ema = disp_to_depth(depth_uncrop_ema, self.opt.min_depth, self.opt.max_depth)

        error = list(self.compute_depth_losses(inputs['depth_gt_uncrop'], depth_uncrop))
        error_teacher = list(self.compute_depth_losses(inputs['depth_gt_uncrop'], depth_uncrop_ema))
        error_ref = list(self.compute_depth_losses(inputs['depth_gt_uncrop'], depth_uncrop_ref))
        error_local = list(self.compute_depth_losses(inputs['depth_gt_uncrop'], depth_uncrop, median_scaling=True))
        error_teacher_local = list(self.compute_depth_losses(inputs['depth_gt_uncrop'], depth_uncrop_ema, median_scaling=True))
        error_local_ref = list(self.compute_depth_losses(inputs['depth_gt_uncrop'], depth_uncrop_ref, median_scaling=True))

        error_unsup = list(self.compute_depth_losses(inputs['depth_gt_uncrop'], reg_depth_unsup, median_scaling=True))
        error_unsup_ref = list(self.compute_depth_losses(inputs['depth_gt_uncrop'], depth_unsup_ref, median_scaling=True))

        for idx, term in enumerate(error):
            error[idx] = term.detach().cpu().numpy()
        for idx, term in enumerate(error_teacher):
            error_teacher[idx] = term.detach().cpu().numpy()
        for idx, term in enumerate(error_local):
            error_local[idx] = term.detach().cpu().numpy()
        for idx, term in enumerate(error_teacher_local):
            error_teacher_local[idx] = term.detach().cpu().numpy()
        for idx, term in enumerate(error_unsup):
            error_unsup[idx] = term.detach().cpu().numpy()
        for idx, term in enumerate(error_ref):
            error_ref[idx] = term.detach().cpu().numpy()
        for idx, term in enumerate(error_local_ref):
            error_local_ref[idx] = term.detach().cpu().numpy()
        for idx, term in enumerate(error_unsup_ref):
            error_unsup_ref[idx] = term.detach().cpu().numpy()

        outputs = {}
        outputs['depth'] = depth_uncrop
        outputs['pseudo_depth_sup'] = pseudo_depth_sup
        outputs['pseudo_depth_unsup'] = pseudo_depth_unsup

        losses = {}
        losses['loss'] = loss_ada.detach().cpu().numpy()
        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = error[i]
        for i, metric in enumerate(self.depth_metric_names_local):
            losses[metric] = error_local[i]

        return error, error_local, error_unsup, outputs, losses,\
               error_ref, error_local_ref, error_unsup_ref, error_teacher, error_teacher_local

    def predict_poses(self, inputs, features, models):
        """Predict poses between input frames for monocular sequences."""
        outputs = {}
        # 只使用存在的帧（仿照 VECTTA，只使用当前帧）
        available_frame_ids = [f_i for f_i in self.opt.frame_ids if ("color_uncrop", f_i, 0) in inputs]
        if len(available_frame_ids) <= 1:
            # 如果只有当前帧，不需要预测 pose
            return outputs
        
        pose_feats = {f_i: (inputs["color_uncrop", f_i, 0]-self.mean)/self.std for f_i in available_frame_ids}

        for f_i in available_frame_ids[1:]:
            if f_i < 0:
                pose_inputs = [pose_feats[f_i], pose_feats[0]]
            else:
                pose_inputs = [pose_feats[0], pose_feats[f_i]]

            pose_inputs = [models["pose_encoder"](torch.cat(pose_inputs, 1))]

            axisangle, translation = models["pose"](pose_inputs)
            outputs[("axisangle", 0, f_i)] = axisangle
            outputs[("translation", 0, f_i)] = translation

            outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        return outputs

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch."""
        # 只使用存在的帧（仿照 VECTTA，只使用当前帧）
        available_frame_ids = [f_i for f_i in self.opt.frame_ids if ("color_uncrop", f_i, 0) in inputs]
        if len(available_frame_ids) <= 1:
            # 如果只有当前帧，不需要生成 warped images
            return
        
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            disp = F.interpolate(
                disp, [self.height, self.width], mode="bilinear", align_corners=False)
            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            source_scale = 0
            for i, frame_id in enumerate(available_frame_ids[1:]):
                if ("cam_T_cam", 0, frame_id) not in outputs:
                    continue
                T = outputs[("cam_T_cam", 0, frame_id)]

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                source_image = inputs[("color_uncrop", frame_id, source_scale)]

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    source_image,
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border")

                outputs[("color_identity", frame_id, scale)] = source_image

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images"""
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_losses_unsup(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch"""
        losses = {}
        total_loss = 0

        # 只使用存在的帧（仿照 VECTTA，只使用当前帧）
        available_frame_ids = [f_i for f_i in self.opt.frame_ids if ("color_uncrop", f_i, 0) in inputs]
        
        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []
            source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color_uncrop", 0, scale)]
            target = inputs[("color_uncrop", 0, source_scale)]

            # 只使用存在的帧
            for frame_id in available_frame_ids[1:]:
                if ("color", frame_id, scale) in outputs:
                    pred = outputs[("color", frame_id, scale)]
                    reprojection_losses.append(self.compute_reprojection_loss(pred, target))
            
            if len(reprojection_losses) > 0:
                reprojection_losses = torch.cat(reprojection_losses, 1)
            else:
                # 如果没有相邻帧，使用 identity loss
                reprojection_losses = torch.zeros_like(target[:, :1, :, :])
            
            identity_reprojection_losses = []
            for frame_id in available_frame_ids[1:]:
                if ("color_uncrop", frame_id, source_scale) in inputs:
                    pred = inputs[("color_uncrop", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))
            
            if len(identity_reprojection_losses) > 0:
                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)
            else:
                identity_reprojection_losses = torch.zeros_like(target[:, :1, :, :])
            identity_reprojection_loss = identity_reprojection_losses
            reprojection_loss = reprojection_losses

            identity_reprojection_loss += torch.randn(
                identity_reprojection_loss.shape).cuda() * 0.00001
            combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            to_optimise, idxs = torch.min(combined, dim=1)
            outputs["identity_selection/{}".format(scale)] = (
                idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss

        return losses

    def compute_losses(self, depth, pseudo_depth_sup, pseudo_depth_unsup):
        """Compute adaptation loss using pseudo labels"""
        mask = pseudo_depth_sup > 1.0
        d = torch.log(depth[mask]) - torch.log(pseudo_depth_sup[mask])
        # Scale invariant
        variance_focus = 0.85
        loss_sup = torch.sqrt((d ** 2).mean() - variance_focus * (d.mean() ** 2)) * 10.0

        mask = pseudo_depth_unsup > 1.0
        d = torch.log(depth[mask]) - torch.log(pseudo_depth_unsup[mask])
        # Scale invariant
        variance_focus = 0.85
        loss_unsup = torch.sqrt((d ** 2).mean() - variance_focus * (d.mean() ** 2)) * 10.0

        loss = loss_sup + loss_unsup
        return loss

    def augment_pseudo(self, inputs, reg_depth_sup, reg_depth_unsup, depth_uncrop_ref):
        """Generate pseudo labels with augmentation"""
        # 确保所有深度图尺寸一致（以 depth_uncrop_ref 为基准）
        target_h, target_w = depth_uncrop_ref.shape[2], depth_uncrop_ref.shape[3]
        
        # 调整 reg_depth_unsup 和 reg_depth_sup 到目标尺寸
        if reg_depth_unsup.shape[2] != target_h or reg_depth_unsup.shape[3] != target_w:
            reg_depth_unsup = F.interpolate(
                reg_depth_unsup, [target_h, target_w], mode="bilinear", align_corners=False)
        if reg_depth_sup.shape[2] != target_h or reg_depth_sup.shape[3] != target_w:
            reg_depth_sup = F.interpolate(
                reg_depth_sup, [target_h, target_w], mode="bilinear", align_corners=False)
        
        # 确保所有张量都在同一设备上
        depth_uncrop_ref = depth_uncrop_ref.to(reg_depth_unsup.device)
        reg_depth_sup = reg_depth_sup.to(reg_depth_unsup.device)
        
        mask0 = depth_uncrop_ref > self.MIN_DEPTH
        mask0[:, :, :int(0.3*depth_uncrop_ref.shape[2]), :] = False
        
        # 确保 mask0 和 reg_depth_unsup 尺寸匹配
        if mask0.shape != reg_depth_unsup.shape:
            # 如果尺寸不匹配，重新创建 mask0
            mask0 = reg_depth_unsup > self.MIN_DEPTH
            mask0[:, :, :int(0.3*reg_depth_unsup.shape[2]), :] = False
        
        scale_factor = torch.median(depth_uncrop_ref[mask0]) / torch.median(reg_depth_unsup[mask0])
        if self.opt.dataset == "kitti":
            gt_height, gt_width = depth_uncrop_ref.shape[2], depth_uncrop_ref.shape[3]
            crop_mask = torch.zeros_like(mask0)
            crop_mask[:, :, int(0.40810811*gt_height):int(0.99189189*gt_height),
                      int(0.03594771*gt_width):int(0.96405229*gt_width)] = 1
            mask0 = crop_mask

        mask = (((depth_uncrop_ref - reg_depth_unsup * scale_factor) ** 2) / depth_uncrop_ref) < self.opt.thres
        reg_depth_unsup = torch.mul(mask.float(), reg_depth_unsup) * scale_factor
        reg_depth_unsup = torch.clamp(reg_depth_unsup, min=self.MIN_DEPTH, max=self.MAX_DEPTH)

        mask = (((reg_depth_sup - depth_uncrop_ref) ** 2) / reg_depth_sup) < self.opt.thres
        mask = mask * mask0

        reg_depth_sup = torch.mul(mask.float(), depth_uncrop_ref)
        reg_depth_sup = torch.clamp(reg_depth_sup, min=self.MIN_DEPTH, max=self.MAX_DEPTH)

        return reg_depth_sup, reg_depth_unsup

    def compute_depth_losses(self, depth_gt, depth_pred, median_scaling=False):
        """Compute depth evaluation metrics"""
        gt_height, gt_width = depth_gt.shape[2], depth_gt.shape[3]
        mask = torch.logical_and(depth_gt > self.MIN_DEPTH, depth_gt < self.MAX_DEPTH)
        depth_pred = F.interpolate(depth_pred, [gt_height, gt_width], mode="bilinear", align_corners=False)

        if self.opt.dataset == "kitti":
            crop_mask = torch.zeros_like(mask)
            crop_mask[:, :, int(0.40810811*gt_height):int(0.99189189*gt_height),
                      int(0.03594771*gt_width):int(0.96405229*gt_width)] = 1
            mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        median_ratio = torch.median(depth_gt)/torch.median(depth_gt)
        if median_scaling:
            median_ratio = torch.median(depth_gt) / torch.median(depth_pred)
            depth_pred *= median_ratio

        depth_gt = torch.clamp(depth_gt, min=self.MIN_DEPTH, max=self.MAX_DEPTH)
        depth_pred = torch.clamp(depth_pred, min=self.MIN_DEPTH, max=self.MAX_DEPTH)

        abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = compute_depth_errors(depth_gt, depth_pred)

        return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3, median_ratio

    def load_model(self, load_path, model_name, models_to_load):
        """Load model(s) from disk"""
        load_path = os.path.expanduser(load_path)

        assert os.path.isdir(load_path), \
            "Cannot find folder {}".format(load_path)
        print("loading model from folder {}".format(load_path))

        for n in models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(load_path, "{}.pth".format(n))
            model_dict = model_name[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model_name[n].load_state_dict(model_dict)


if __name__ == "__main__":
    options = MonodepthOptions()
    
    # 添加 ada-depth 需要的参数
    options.parser.add_argument("--reg_path",
                                type=str,
                                help="reg models to load",
                                default=None)
    options.parser.add_argument("--thres",
                                type=float,
                                help="using a threshold value to select better pixels for pseudo gt",
                                default=0.4)
    
    opts = options.parse()
    adapt = AdaptSQLdepth(opts)
    adapt.run_adapt()

