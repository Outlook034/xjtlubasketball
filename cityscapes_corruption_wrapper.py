"""
Cityscapes corruption wrapper - 修复目录结构问题
确保生成的目录结构为: corruption_name/severity_X/city_name/image_file.png
"""
import os
from PIL import Image
import torch
from torch.utils import data


def create_corruption_with_city_structure(dataloader, save_path, corruption_name, severity_levels, corruption_func):
    """
    包装函数：创建扰动并保存到正确的目录结构
    
    Args:
        dataloader: 数据加载器，返回 (image_tensor, filename, city_name)
        save_path: 保存根目录
        corruption_name: 扰动类型名称（如 'brightness'）
        severity_levels: 严重程度列表 [1, 2, 3, 4, 5]
        corruption_func: 扰动函数，接受 (image, severity) 返回扰动后的图像
    """
    corruption_dir = os.path.join(save_path, corruption_name)
    os.makedirs(corruption_dir, exist_ok=True)
    
    # 为每个严重程度创建目录
    for severity in severity_levels:
        severity_dir = os.path.join(corruption_dir, f'severity_{severity}')
        os.makedirs(severity_dir, exist_ok=True)
    
    print(f"Processing {len(dataloader)} images for {corruption_name}...")
    
    for batch_idx, batch in enumerate(dataloader):
        if len(batch) == 3:
            image_tensor, filename, city_name = batch
        else:
            # 兼容旧版本（只有2个返回值）
            image_tensor, filename = batch
            # 从文件名提取城市名
            city_name = filename[0].split('_')[0] if isinstance(filename, (list, tuple)) else filename.split('_')[0]
        
        # 转换为PIL Image
        if isinstance(image_tensor, torch.Tensor):
            if image_tensor.dim() == 3:
                # [H, W, C] -> PIL Image
                img_np = image_tensor.numpy().astype('uint8')
                img = Image.fromarray(img_np)
            elif image_tensor.dim() == 4:
                # [B, H, W, C] -> 取第一个
                img_np = image_tensor[0].numpy().astype('uint8')
                img = Image.fromarray(img_np)
            else:
                continue
        else:
            img = image_tensor
        
        # 获取文件名
        if isinstance(filename, (list, tuple)):
            filename = filename[0]
        if isinstance(city_name, (list, tuple)):
            city_name = city_name[0]
        
        # 为每个严重程度生成扰动图像
        for severity in severity_levels:
            # 应用扰动
            corrupted_img = corruption_func(img, severity)
            
            # 创建城市目录
            severity_dir = os.path.join(corruption_dir, f'severity_{severity}')
            city_dir = os.path.join(severity_dir, city_name)
            os.makedirs(city_dir, exist_ok=True)
            
            # 保存图像
            save_path_img = os.path.join(city_dir, filename)
            corrupted_img.save(save_path_img)
        
        if (batch_idx + 1) % 100 == 0:
            print(f"  Processed {batch_idx + 1}/{len(dataloader)} images...")
    
    print(f"Completed {corruption_name} corruptions!")


def copy_clean_with_city_structure(dataloader, save_path):
    """
    复制干净图像到正确的目录结构
    """
    clean_dir = os.path.join(save_path, 'clean')
    os.makedirs(clean_dir, exist_ok=True)
    
    print(f"Copying {len(dataloader)} clean images...")
    
    for batch_idx, batch in enumerate(dataloader):
        if len(batch) == 3:
            image_tensor, filename, city_name = batch
        else:
            image_tensor, filename = batch
            city_name = filename[0].split('_')[0] if isinstance(filename, (list, tuple)) else filename.split('_')[0]
        
        # 转换为PIL Image
        if isinstance(image_tensor, torch.Tensor):
            if image_tensor.dim() == 3:
                img_np = image_tensor.numpy().astype('uint8')
                img = Image.fromarray(img_np)
            elif image_tensor.dim() == 4:
                img_np = image_tensor[0].numpy().astype('uint8')
                img = Image.fromarray(img_np)
            else:
                continue
        else:
            img = image_tensor
        
        # 获取文件名
        if isinstance(filename, (list, tuple)):
            filename = filename[0]
        if isinstance(city_name, (list, tuple)):
            city_name = city_name[0]
        
        # 创建城市目录
        city_dir = os.path.join(clean_dir, city_name)
        os.makedirs(city_dir, exist_ok=True)
        
        # 保存图像
        save_path_img = os.path.join(city_dir, filename)
        img.save(save_path_img)
        
        if (batch_idx + 1) % 100 == 0:
            print(f"  Copied {batch_idx + 1}/{len(dataloader)} images...")
    
    print("Completed copying clean images!")






























