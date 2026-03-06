import argparse
import os
import random
import glob
import time
from PIL import Image
import numpy as np
import torch
from torch.utils import data
from datetime import timedelta

# corruptions: weather & lighting
import cityscapes_corruptions_utils as citycorr_utils
try:
    from imagecorruptions import corrupt, get_corruption_names
except ImportError:
    print("Error: 'imagecorruptions' library not found. Please install it via 'pip install imagecorruptions'.")
    corrupt = None
    get_corruption_names = lambda: []

def pil_collate_fn(batch):
    """自定义collate函数，用于处理包含PIL Image的batch"""
    if len(batch) == 1:
        if batch[0] is None:
            return None
        return batch[0]
    # 过滤掉 None
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    images, filenames, city_names = zip(*batch)
    return list(images), list(filenames), list(city_names)

class CityscapesImageDataset(data.Dataset):
    """根据 test_list 寻找数据的 Dataset"""
    def __init__(self, image_root, suffix='_leftImg8bit.png', test_list=None, folder_name='leftImg8bit_sequence', split='test'):
        if test_list is None or not os.path.isfile(test_list):
            raise ValueError(f"test_list is required and must be a valid file. Got: {test_list}")
        
        with open(test_list, 'r') as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        
        image_paths = []
        print(f"Reading test list from {test_list}...")
        for ln in lines:
            parts = ln.split()
            if len(parts) < 2:
                continue
            city, frame_name = parts[0], parts[1]
            
            # 构造几种可能的路径结构进行匹配
            candidates = [
                os.path.join(image_root, folder_name, split, city, frame_name + suffix),
                os.path.join(image_root, city, frame_name + suffix),
                os.path.join(image_root, frame_name + suffix)
            ]
            
            found = False
            for cand in candidates:
                if os.path.isfile(cand):
                    image_paths.append(cand)
                    found = True
                    break
            
            if not found:
                # 如果没找到，脚本会提醒但不再递归，避免引入错误文件
                print(f"Warning: Image not found for entry '{city} {frame_name}'.")

        self.image_paths = sorted(image_paths)
        if not self.image_paths:
            raise ValueError(f"No images found for the given test list under {image_root}")
        print(f"Successfully located {len(self.image_paths)} images.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            file_name = os.path.basename(img_path)
            # 提取城市名用于目录结构
            city = file_name.split('_')[0]
            return img, file_name, city
        except Exception as e:
            print(f"\nError loading {img_path}: {e}")
            return None

def get_args():
    parser = argparse.ArgumentParser(description='Create Cityscapes-C with Countdown (Fixed)')
    parser.add_argument('--image_root', type=str, required=True, help="Cityscapes leftImg8bit root")
    parser.add_argument('--test_list', type=str, required=True, help="Path to test_files.txt")
    parser.add_argument('--save_path', type=str, default="data_cityscapes_corruptions")
    parser.add_argument('--severity_levels', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    
    # 扰动开关
    parser.add_argument('--if_brightness', action='store_true')
    parser.add_argument('--if_dark', action='store_true')
    parser.add_argument('--if_fog', action='store_true')
    parser.add_argument('--if_frost', action='store_true')
    parser.add_argument('--if_snow', action='store_true')
    parser.add_argument('--if_contrast', action='store_true')
    parser.add_argument('--if_defocus_blur', action='store_true')
    parser.add_argument('--if_glass_blur', action='store_true')
    parser.add_argument('--if_motion_blur', action='store_true')
    parser.add_argument('--if_zoom_blur', action='store_true')
    parser.add_argument('--if_elastic', action='store_true')
    parser.add_argument('--if_color_quant', action='store_true')
    parser.add_argument('--if_gaussian_noise', action='store_true')
    parser.add_argument('--if_impulse_noise', action='store_true')
    parser.add_argument('--if_shot_noise', action='store_true')
    parser.add_argument('--if_iso_noise', action='store_true')
    parser.add_argument('--if_pixelate', action='store_true')
    parser.add_argument('--if_jpeg', action='store_true')
    
    return parser.parse_args()

def main():
    args = get_args()
    
    dataset = CityscapesImageDataset(image_root=args.image_root, test_list=args.test_list)
    dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=pil_collate_fn)

    corruption_list = [
        'brightness', 'dark', 'fog', 'frost', 'snow', 'contrast',
        'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
        'elastic', 'color_quant', 'gaussian_noise', 'impulse_noise',
        'shot_noise', 'iso_noise', 'pixelate', 'jpeg'
    ]
    
    active_corruptions = [c for c in corruption_list if getattr(args, f'if_{c}')]
    if not active_corruptions:
        print("No corruptions selected. Use --if_xxx flags.")
        return

    # 映射表，确保名称与库一致
    all_supported = get_corruption_names()
    corruption_map = {
        'brightness': 'brightness',
        'dark': 'low_light', # 使用 utils.low_light
        'fog': 'fog',
        'frost': 'frost',
        'snow': 'snow',
        'contrast': 'contrast',
        'defocus_blur': 'defocus_blur',
        'glass_blur': 'glass_blur',
        'motion_blur': 'motion_blur',
        'zoom_blur': 'zoom_blur',
        'elastic': 'elastic_transform',
        'color_quant': 'color_quantization' if 'color_quantization' in all_supported else 'color_quant',
        'gaussian_noise': 'gaussian_noise',
        'impulse_noise': 'impulse_noise',
        'shot_noise': 'shot_noise',
        'iso_noise': 'iso_noise',
        'pixelate': 'pixelate',
        'jpeg': 'jpeg_compression'
    }

    total_tasks = len(active_corruptions) * len(dataset) * len(args.severity_levels)
    completed_tasks = 0
    start_time = time.time()

    print(f"\nStarting Cityscapes-C generation.")
    print(f"Total tasks: {total_tasks} ({len(active_corruptions)} types x {len(dataset)} imgs x {len(args.severity_levels)} sevs)")

    for c_name in active_corruptions:
        c_key = corruption_map.get(c_name, c_name)
        print(f"\n>>> Processing Corruption: {c_name} (key: {c_key})")
        
        for batch_idx, batch in enumerate(dataloader):
            if batch is None:
                continue
            
            img, filename, city_name = batch
            img_np = np.array(img)
            
            for sev in args.severity_levels:
                try:
                    if c_name == 'dark':
                        corr_out = citycorr_utils.low_light(img_np, severity=sev)
                    elif c_name == 'glass_blur':
                        # imagecorruptions glass_blur may break due to skimage API changes (multichannel/channel_axis)
                        # so we use a cv2/numpy fallback implementation here.
                        corr_out = citycorr_utils.glass_blur_cv2(img_np, severity=sev)
                    elif c_name == 'color_quant':
                        corr_out = citycorr_utils.color_quant(img, severity=sev)
                    elif c_name == 'iso_noise':
                        corr_out = citycorr_utils.iso_noise(img_np, severity=sev)
                    else:
                        if corrupt is None:
                            raise RuntimeError("imagecorruptions library not installed")
                        corr_out = corrupt(img_np, corruption_name=c_key, severity=sev)
                    
                    # 构造输出路径: save_path/corruption/severity_X/city/xxx.png
                    out_dir = os.path.join(args.save_path, c_name, f'severity_{sev}', city_name)
                    os.makedirs(out_dir, exist_ok=True)

                    # corr_out may be a PIL.Image (custom utils) or numpy array (imagecorruptions)
                    if isinstance(corr_out, Image.Image):
                        corr_img = corr_out
                    else:
                        corr_img = Image.fromarray(np.uint8(corr_out))

                    corr_img.save(os.path.join(out_dir, filename))
                except Exception as e:
                    print(f"\n[Error] {c_name} sev {sev} on {filename}: {e}")

                # 倒计时
                completed_tasks += 1
                elapsed = time.time() - start_time
                avg_per_task = elapsed / completed_tasks
                remaining = total_tasks - completed_tasks
                eta = timedelta(seconds=int(remaining * avg_per_task))
                
                if completed_tasks % 5 == 0:
                    print(f"\rProgress: [{completed_tasks}/{total_tasks}] {completed_tasks/total_tasks:.1%} | "
                          f"ETA: {str(eta)} | Curr: {c_name} sev {sev} | Img: {batch_idx}", end="")

    print(f"\n\nAll Tasks Completed! Total Time: {timedelta(seconds=int(time.time() - start_time))}")

if __name__ == '__main__':
    main()
