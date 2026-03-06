import argparse
import os
import random
import glob
from PIL import Image  # Make sure PIL.Image is imported
import numpy as np
import torch
from torch.utils import data

# corruptions: weather & lighting
import utils as citycorr_utils
# corruptions: sensor & movement
_ = citycorr_utils  # alias used for access
# corruptions: data & processing
_ = citycorr_utils
# copy clean
_ = citycorr_utils

# 导入修复保存路径的工具函数
from imagecorruptions import corrupt


def pil_collate_fn(batch):
    """自定义collate函数，用于处理包含PIL Image的batch"""
    # batch是一个列表，每个元素是 (img, filename, city_name)
    # 由于batch_size=1，我们直接返回第一个元素
    if len(batch) == 1:
        return batch[0]
    # 如果batch_size>1，返回列表
    images, filenames, city_names = zip(*batch)
    return list(images), list(filenames), list(city_names)


class CityscapesImageDataset(data.Dataset):
    """Cityscapes image folder dataset for corruption generation.

    - Recursively scans `image_root` for files matching `*{suffix}` (default: `_leftImg8bit.png`).
    - 保持原始尺寸，不resize（让评估时自己处理裁剪和resize）.
    - Returns PIL Image and filename string.
    """

    def __init__(self, image_root, H, W, suffix='_leftImg8bit.png', test_list=None, folder_name='leftImg8bit_sequence', split='test'):
        """
        image_root: Cityscapes根目录（包含 leftImg8bit(_sequence)/test/...）或直接指向该子目录
        test_list: splits 文件，如 test_files.txt，行内格式类似 "aachen aachen_000000 4"（必需）
        folder_name: 'leftImg8bit' 或 'leftImg8bit_sequence'
        split: 'test'（默认）
        """
        if test_list is None or not os.path.isfile(test_list):
            raise ValueError(f"test_list is required and must be a valid file. Got: {test_list}")
        
        with open(test_list, 'r') as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        image_paths = []
        not_found_count = 0
        # 允许 image_root 既可以是 Cityscapes 根，也可以是具体的 leftImg8bit(_sequence)/split 目录
        for ln in lines:
            parts = ln.split()
            if len(parts) < 2:
                continue
            city, frame_name = parts[0], parts[1]
            rel_dir = os.path.join(folder_name, split, city)
            candidate_1 = os.path.join(image_root, rel_dir, frame_name + suffix)
            candidate_2 = os.path.join(image_root, city, frame_name + suffix)
            if os.path.isfile(candidate_1):
                image_paths.append(candidate_1)
            elif os.path.isfile(candidate_2):
                image_paths.append(candidate_2)
            else:
                # 回退：在 image_root 下递归查找该文件名
                found = glob.glob(os.path.join(image_root, '**', frame_name + suffix), recursive=True)
                if found:
                    image_paths.append(found[0])
                else:
                    not_found_count += 1
                    if not_found_count <= 5:  # 只显示前5个警告
                        print(f"Warning: Not found for list entry '{city} {frame_name}' -> '{frame_name + suffix}'")
        if not_found_count > 5:
            print(f"Warning: ... and {not_found_count - 5} more images not found.")
        self.image_paths = sorted(image_paths)
        self.H = H
        self.W = W
        if not self.image_paths:
            raise ValueError(f"No images found from test_list '{test_list}'. Please check the file paths.")
        else:
            print(f"Found {len(self.image_paths)} images from test_list '{test_list}' (out of {len(lines)} entries)")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # 返回一个占位图像（原始尺寸）
            return Image.new('RGB', (2048, 1024), color='black'), "error_path.png", "unknown"

        # 不resize，保持原始尺寸，让评估时自己处理
        if idx == 0:
            print(f"[Info] Keeping original image size: {img.size} (W x H), no resize")

        file_name = os.path.basename(img_path)
        # 从路径中提取城市名：leftImg8bit/test/berlin/xxx.png -> berlin
        path_parts = img_path.replace('\\', '/').split('/')
        city = "unknown"
        for i, part in enumerate(path_parts):
            if part in ['test', 'train', 'val'] and i + 1 < len(path_parts):
                city = path_parts[i + 1]
                break
        # 如果没找到，尝试从文件名提取（假设文件名格式为 city_xxx.png）
        if city == "unknown" and '_' in file_name:
            city = file_name.split('_')[0]
        
        # 返回PIL Image（保持原始尺寸），不resize
        return img, file_name, city


def get_args():
    parser = argparse.ArgumentParser(description='Create Cityscapes Corruptions')

    # general configurations
    parser.add_argument('--image_root', type=str,
                        help="Path to Cityscapes image root, e.g., .../leftImg8bit_sequence/test or .../leftImg8bit/test",
                        default="path/to/cityscapes/leftImg8bit_sequence/test")
    parser.add_argument('--test_list', type=str, required=True,
                        help="Path to Cityscapes test list (e.g., splits/cityscapes/test_files.txt) [REQUIRED]")
    parser.add_argument('--folder_name', type=str,
                        help="Cityscapes folder name (leftImg8bit or leftImg8bit_sequence)",
                        default='leftImg8bit_sequence')
    parser.add_argument('--split', type=str, help="Cityscapes split", default='test')
    parser.add_argument('--suffix', type=str,
                        help="Filename suffix to match", default="_leftImg8bit.png")
    parser.add_argument('--H', type=int, help='height for the image.', default=320)
    parser.add_argument('--W', type=int, help='width for the image.', default=1024)
    parser.add_argument('--save_path', type=str,
                        help="Root path for saving corrupted images.", default="data_cityscapes_corruptions")
    parser.add_argument('--seed', type=int, help='random seed.', default=42)
    parser.add_argument('--deterministic', action='store_true',
                        help='whether to set deterministic options for CUDNN backend.')

    # corruption configurations
    parser.add_argument('--if_brightness', action='store_true', help="create 'brightness' corruptions")
    parser.add_argument('--if_dark', action='store_true', help="create 'dark' corruptions")
    parser.add_argument('--if_fog', action='store_true', help="create 'fog' corruptions")
    parser.add_argument('--if_frost', action='store_true', help="create 'frost' corruptions")
    parser.add_argument('--if_snow', action='store_true', help="create 'snow' corruptions")
    parser.add_argument('--if_contrast', action='store_true', help="create 'contrast' corruptions")
    parser.add_argument('--if_defocus_blur', action='store_true', help="create 'defocus_blur' corruptions")
    parser.add_argument('--if_glass_blur', action='store_true', help="create 'glass_blur' corruptions")
    parser.add_argument('--if_motion_blur', action='store_true', help="create 'motion_blur' corruptions")
    parser.add_argument('--if_zoom_blur', action='store_true', help="create 'zoom_blur' corruptions")
    parser.add_argument('--if_elastic', action='store_true', help="create 'elastic' corruptions")
    parser.add_argument('--if_color_quant', action='store_true', help="create 'color_quant' corruptions")
    parser.add_argument('--if_gaussian_noise', action='store_true', help="create 'gaussian_noise' corruptions")
    parser.add_argument('--if_impulse_noise', action='store_true', help="create 'impulse_noise' corruptions")
    parser.add_argument('--if_shot_noise', action='store_true', help="create 'shot_noise' corruptions")
    parser.add_argument('--if_iso_noise', action='store_true', help="create 'iso_noise' corruptions")
    parser.add_argument('--if_pixelate', action='store_true', help="create 'pixelate' corruptions")
    parser.add_argument('--if_jpeg', action='store_true', help="create 'jpeg_compression' corruptions")
    parser.add_argument('--if_copy_clean', action='store_true', help="copy 'clean' images")

    # corruption severity levels
    parser.add_argument('--severity_levels', type=int, nargs='+',
                        help="severity levels to be applied (e.g., 1 2 3 4 5).", default=[1, 2, 3, 4, 5])

    return parser.parse_args()


def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    args = get_args()

    print(f"[Debug] Args received: H={args.H}, W={args.W}")
    if args.H < 32 or args.W < 32:
        print(f"Error: Image height (H={args.H}) and width (W={args.W}) must both be >= 32.")
        return

    # 处理severity_levels参数
    if not args.severity_levels:
        args.severity_levels = [1, 2, 3, 4, 5]  # 默认值
    else:
        # 确保是整数列表，且在有效范围内
        args.severity_levels = [int(s) for s in args.severity_levels if 1 <= int(s) <= 5]
        if not args.severity_levels:
            print("Warning: No valid severity levels (must be 1-5). Using default [1,2,3,4,5]")
            args.severity_levels = [1, 2, 3, 4, 5]
    
    print(f"[Debug] Severity levels: {args.severity_levels}")

    if args.seed is not None:
        print(f'Set random seed to {args.seed}, deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)

    print(f"Loading Cityscapes images from: {args.image_root}")
    print(f"Using test list: {args.test_list} (only processing images listed in this file)")
    dataset = CityscapesImageDataset(
        image_root=args.image_root,
        H=args.H,
        W=args.W,
        suffix=args.suffix,
        test_list=args.test_list,
        folder_name=args.folder_name,
        split=args.split,
    )
    if len(dataset) == 0:
        print("No images found. Exiting.")
        return

    dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False, collate_fn=pil_collate_fn)
    print("Dataset loaded successfully.\n")

    os.makedirs(args.save_path, exist_ok=True)

    # 定义扰动类型映射（imagecorruptions库使用的名称）
    corruption_map = {
        'brightness': 'brightness',
        'dark': 'low_light',  # 注意：可能需要特殊处理
        'fog': 'fog',
        'frost': 'frost',
        'snow': 'snow',
        'contrast': 'contrast',
        'defocus_blur': 'defocus_blur',
        'glass_blur': 'glass_blur',
        'motion_blur': 'motion_blur',
        'zoom_blur': 'zoom_blur',
        'elastic': 'elastic_transform',
        'color_quant': 'color_quantization',
        'gaussian_noise': 'gaussian_noise',
        'impulse_noise': 'impulse_noise',
        'shot_noise': 'shot_noise',
        'iso_noise': 'iso_noise',
        'pixelate': 'pixelate',
        'jpeg': 'jpeg_compression',
    }

    def process_corruption(corruption_name):
        """处理单个扰动类型"""
        # 获取imagecorruptions库使用的扰动名称
        corruption_key = corruption_map.get(corruption_name, corruption_name)
        
        print(f"Creating '{corruption_name}' corruptions ...")
        processed = 0
        for batch_idx, batch in enumerate(dataloader):
            # 由于使用了pil_collate_fn且batch_size=1，batch直接是 (img, filename, city_name)
            if isinstance(batch, tuple) and len(batch) == 3:
                img, filename, city_name = batch
            elif isinstance(batch, tuple) and len(batch) == 2:
                img, filename = batch
                # 从文件名提取城市名
                if isinstance(filename, (list, tuple)):
                    city_name = filename[0].split('_')[0] if filename[0] else "unknown"
                    filename = filename[0]
                else:
                    city_name = filename.split('_')[0] if filename else "unknown"
            else:
                # 兼容旧格式（如果collate返回的是列表）
                if len(batch) == 3:
                    img, filename, city_name = batch[0], batch[1], batch[2]
                else:
                    continue
            
            # 确保img是PIL Image
            if not isinstance(img, Image.Image):
                if isinstance(img, torch.Tensor):
                    # 兼容旧代码（如果返回的是tensor）
                    if img.dim() == 3:
                        img_np = img.numpy().astype('uint8')
                        img = Image.fromarray(img_np)
                    elif img.dim() == 4:
                        img_np = img[0].numpy().astype('uint8')
                        img = Image.fromarray(img_np)
                    else:
                        continue
                else:
                    continue
            
            # 处理城市名和文件名（如果是list或tuple）
            if isinstance(city_name, (list, tuple)):
                city_name = city_name[0] if city_name else "unknown"
            if isinstance(filename, (list, tuple)):
                filename = filename[0] if filename else "unknown.png"
            
            # 为每个严重程度生成扰动
            for severity in args.severity_levels:
                try:
                    # 对于dark，使用utils的low_light函数
                    if corruption_name == 'dark':
                        corrupted_np = citycorr_utils.low_light(np.array(img), severity=severity)
                        corrupted_img = Image.fromarray(corrupted_np)
                    else:
                        # 使用imagecorruptions库应用扰动
                        img_np = np.array(img)
                        corrupted_img_np = corrupt(img_np, corruption_name=corruption_key, severity=severity)
                        corrupted_img = Image.fromarray(corrupted_img_np)
                    
                    # 创建目录结构: save_path/corruption_name/severity_X/city_name/
                    corruption_dir = os.path.join(args.save_path, corruption_name)
                    severity_dir = os.path.join(corruption_dir, f'severity_{severity}')
                    city_dir = os.path.join(severity_dir, city_name)
                    os.makedirs(city_dir, exist_ok=True)
                    
                    # 保存图像
                    save_path_img = os.path.join(city_dir, filename)
                    corrupted_img.save(save_path_img)
                except Exception as e:
                    print(f"Error processing {filename} with {corruption_name} severity {severity}: {e}")
                    continue
            
            processed += 1
            if (batch_idx + 1) % 100 == 0:
                print(f"  Processed {batch_idx + 1}/{len(dataloader)} images...")
        
        print(f"Successful! Processed {processed} images, saved to: '{os.path.join(args.save_path, corruption_name)}'\n")

    if args.if_brightness:
        process_corruption('brightness')

    if args.if_dark:
        process_corruption('dark')

    if args.if_fog:
        process_corruption('fog', 'fog')

    if args.if_frost:
        process_corruption('frost', 'frost')

    if args.if_snow:
        process_corruption('snow', 'snow')

    if args.if_contrast:
        process_corruption('contrast', 'contrast')

    if args.if_defocus_blur:
        process_corruption('defocus_blur', 'defocus_blur')

    if args.if_glass_blur:
        process_corruption('glass_blur', 'glass_blur')

    if args.if_motion_blur:
        process_corruption('motion_blur', 'motion_blur')

    if args.if_zoom_blur:
        process_corruption('zoom_blur', 'zoom_blur')

    if args.if_elastic:
        process_corruption('elastic_transform')

    if args.if_color_quant:
        process_corruption('color_quantization')

    if args.if_gaussian_noise:
        process_corruption('gaussian_noise', 'gaussian_noise')

    if args.if_impulse_noise:
        process_corruption('impulse_noise', 'impulse_noise')

    if args.if_shot_noise:
        process_corruption('shot_noise', 'shot_noise')

    if args.if_iso_noise:
        process_corruption('iso_noise', 'iso_noise')

    if args.if_pixelate:
        process_corruption('pixelate', 'pixelate')

    if args.if_jpeg:
        process_corruption('jpeg_compression', 'jpeg_compression')

    if args.if_copy_clean:
        print("Copying clean set ...")
        processed = 0
        for batch_idx, batch in enumerate(dataloader):
            # 由于使用了pil_collate_fn且batch_size=1，batch直接是 (img, filename, city_name)
            if isinstance(batch, tuple) and len(batch) == 3:
                img, filename, city_name = batch
            elif isinstance(batch, tuple) and len(batch) == 2:
                img, filename = batch
                # 从文件名提取城市名
                if isinstance(filename, (list, tuple)):
                    city_name = filename[0].split('_')[0] if filename[0] else "unknown"
                    filename = filename[0]
                else:
                    city_name = filename.split('_')[0] if filename else "unknown"
            else:
                # 兼容旧格式（如果collate返回的是列表）
                if len(batch) == 3:
                    img, filename, city_name = batch[0], batch[1], batch[2]
                else:
                    continue
            
            # 确保img是PIL Image
            if not isinstance(img, Image.Image):
                if isinstance(img, torch.Tensor):
                    # 兼容旧代码（如果返回的是tensor）
                    if img.dim() == 3:
                        img_np = img.numpy().astype('uint8')
                        img = Image.fromarray(img_np)
                    elif img.dim() == 4:
                        img_np = img[0].numpy().astype('uint8')
                        img = Image.fromarray(img_np)
                    else:
                        continue
                else:
                    continue
            
            # 处理城市名和文件名（如果是list或tuple）
            if isinstance(city_name, (list, tuple)):
                city_name = city_name[0] if city_name else "unknown"
            if isinstance(filename, (list, tuple)):
                filename = filename[0] if filename else "unknown.png"
            
            # 创建目录结构: save_path/clean/city_name/
            clean_dir = os.path.join(args.save_path, 'clean')
            city_dir = os.path.join(clean_dir, city_name)
            os.makedirs(city_dir, exist_ok=True)
            
            # 保存图像
            save_path_img = os.path.join(city_dir, filename)
            img.save(save_path_img)
            
            processed += 1
            if (batch_idx + 1) % 100 == 0:
                print(f"  Copied {batch_idx + 1}/{len(dataloader)} images...")
        print(f"Successful! Copied {processed} clean images to: '{os.path.join(args.save_path, 'clean')}'\n")


if __name__ == '__main__':
    main()


