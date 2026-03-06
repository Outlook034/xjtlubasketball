"""
修复Cityscapes-C保存路径的工具函数
确保目录结构为: corruption_name/severity_X/city_name/image_file.png
"""
import os
from PIL import Image
import numpy as np
from imagecorruptions import corrupt


def apply_corruption_and_save(image, filename, city_name, corruption_name, severity, save_base_path):
    """
    应用扰动并保存到正确的目录结构
    
    Args:
        image: PIL Image对象
        filename: 图像文件名
        city_name: 城市名称
        corruption_name: 扰动类型（如 'brightness'）
        severity: 严重程度 (1-5)
        save_base_path: 保存根目录
    """
    # 应用扰动
    img_np = np.array(image)
    corrupted_img_np = corrupt(img_np, corruption_name=corruption_name, severity=severity)
    corrupted_img = Image.fromarray(corrupted_img_np)
    
    # 创建目录结构: save_base_path/corruption_name/severity_X/city_name/
    corruption_dir = os.path.join(save_base_path, corruption_name)
    severity_dir = os.path.join(corruption_dir, f'severity_{severity}')
    city_dir = os.path.join(severity_dir, city_name)
    os.makedirs(city_dir, exist_ok=True)
    
    # 保存图像
    save_path = os.path.join(city_dir, filename)
    corrupted_img.save(save_path)
    
    return save_path


def save_clean_image(image, filename, city_name, save_base_path):
    """
    保存干净图像到正确的目录结构
    
    Args:
        image: PIL Image对象
        filename: 图像文件名
        city_name: 城市名称
        save_base_path: 保存根目录
    """
    # 创建目录结构: save_base_path/clean/city_name/
    clean_dir = os.path.join(save_base_path, 'clean')
    city_dir = os.path.join(clean_dir, city_name)
    os.makedirs(city_dir, exist_ok=True)
    
    # 保存图像
    save_path = os.path.join(city_dir, filename)
    image.save(save_path)
    
    return save_path






























