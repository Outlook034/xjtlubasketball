# Cityscapes-C 扰动测试集生成指南

## 一、Cityscapes测试集说明

### 1.1 测试集文件列表

Cityscapes测试集文件列表位于：`SfMNeXt-Impl/splits/cityscapes/test_files.txt`

**文件格式**：
```
aachen aachen_000000 4
aachen aachen_000001 4
aachen aachen_000002 4
...
```

每行格式：`城市名 帧名称 帧号`

### 1.2 测试集目录结构

Cityscapes测试集的标准目录结构：
```
cityscapes/
├── leftImg8bit_sequence/
│   └── test/
│       ├── aachen/
│       │   ├── aachen_000000_leftImg8bit.png
│       │   ├── aachen_000001_leftImg8bit.png
│       │   └── ...
│       ├── bochum/
│       └── ...
├── camera_trainvaltest/
│   └── camera/
│       └── test/
│           ├── aachen/
│           │   ├── aachen_000000_camera.json
│           │   └── ...
│           └── ...
└── ...
```

### 1.3 测试集统计

- **原始分辨率**: 2048×1024
- **裁剪后分辨率**: 2048×768 (裁剪掉底部25%)
- **测试集大小**: 约500张图像（具体数量取决于test_files.txt）

---

## 二、Cityscapes-C 扰动类型

Cityscapes-C包含以下19种扰动类型，分为5个类别：

### 2.1 天气和光照扰动 (Weather & Lighting)
1. **brightness** - 亮度变化
2. **dark** - 暗化
3. **fog** - 雾化
4. **frost** - 霜冻
5. **snow** - 降雪
6. **contrast** - 对比度变化

### 2.2 模糊扰动 (Blur)
7. **defocus_blur** - 散焦模糊
8. **glass_blur** - 玻璃模糊
9. **motion_blur** - 运动模糊
10. **zoom_blur** - 缩放模糊

### 2.3 数字扰动 (Digital)
11. **elastic** - 弹性变形
12. **color_quant** - 颜色量化
13. **gaussian_noise** - 高斯噪声
14. **impulse_noise** - 脉冲噪声
15. **shot_noise** - 散粒噪声
16. **iso_noise** - ISO噪声
17. **pixelate** - 像素化
18. **jpeg_compression** - JPEG压缩

### 2.4 干净图像 (Clean)
19. **clean** - 原始干净图像（作为baseline）

---

## 三、生成Cityscapes-C的步骤

### 3.1 准备环境

确保已安装必要的依赖：
```bash
pip install torch torchvision pillow numpy
```

### 3.2 使用creat_corruption.py生成扰动

**基本命令格式**：
```bash
python creat_corruption.py \
    --image_root /path/to/cityscapes/leftImg8bit_sequence/test \
    --test_list SfMNeXt-Impl/splits/cityscapes/test_files.txt \
    --folder_name leftImg8bit_sequence \
    --split test \
    --H 320 \
    --W 1024 \
    --save_path data_cityscapes_corruptions \
    --severity_levels [1,2,3,4,5] \
    --if_[corruption_name]
```

### 3.3 生成所有扰动类型

**方法1: 逐个生成**
```bash
# 天气和光照扰动
python creat_corruption.py --image_root ... --if_brightness --severity_levels [1,2,3,4,5]
python creat_corruption.py --image_root ... --if_dark --severity_levels [1,2,3,4,5]
python creat_corruption.py --image_root ... --if_fog --severity_levels [1,2,3,4,5]
python creat_corruption.py --image_root ... --if_frost --severity_levels [1,2,3,4,5]
python creat_corruption.py --image_root ... --if_snow --severity_levels [1,2,3,4,5]
python creat_corruption.py --image_root ... --if_contrast --severity_levels [1,2,3,4,5]

# 模糊扰动
python creat_corruption.py --image_root ... --if_defocus_blur --severity_levels [1,2,3,4,5]
python creat_corruption.py --image_root ... --if_glass_blur --severity_levels [1,2,3,4,5]
python creat_corruption.py --image_root ... --if_motion_blur --severity_levels [1,2,3,4,5]
python creat_corruption.py --image_root ... --if_zoom_blur --severity_levels [1,2,3,4,5]

# 数字扰动
python creat_corruption.py --image_root ... --if_elastic --severity_levels [1,2,3,4,5]
python creat_corruption.py --image_root ... --if_color_quant --severity_levels [1,2,3,4,5]
python creat_corruption.py --image_root ... --if_gaussian_noise --severity_levels [1,2,3,4,5]
python creat_corruption.py --image_root ... --if_impulse_noise --severity_levels [1,2,3,4,5]
python creat_corruption.py --image_root ... --if_shot_noise --severity_levels [1,2,3,4,5]
python creat_corruption.py --image_root ... --if_iso_noise --severity_levels [1,2,3,4,5]
python creat_corruption.py --image_root ... --if_pixelate --severity_levels [1,2,3,4,5]
python creat_corruption.py --image_root ... --if_jpeg --severity_levels [1,2,3,4,5]

# 干净图像
python creat_corruption.py --image_root ... --if_copy_clean
```

**方法2: 批量生成脚本**

创建 `generate_cityscapes_c.sh`：
```bash
#!/bin/bash

IMAGE_ROOT="/path/to/cityscapes/leftImg8bit_sequence/test"
TEST_LIST="SfMNeXt-Impl/splits/cityscapes/test_files.txt"
SAVE_PATH="data_cityscapes_corruptions"
H=320
W=1024
SEVERITY="[1,2,3,4,5]"

# 所有扰动类型
CORRUPTIONS=(
    "brightness" "dark" "fog" "frost" "snow" "contrast"
    "defocus_blur" "glass_blur" "motion_blur" "zoom_blur"
    "elastic" "color_quant" "gaussian_noise" "impulse_noise"
    "shot_noise" "iso_noise" "pixelate" "jpeg"
)

for corr in "${CORRUPTIONS[@]}"; do
    echo "Generating $corr corruptions..."
    python creat_corruption.py \
        --image_root $IMAGE_ROOT \
        --test_list $TEST_LIST \
        --folder_name leftImg8bit_sequence \
        --split test \
        --H $H \
        --W $W \
        --save_path $SAVE_PATH \
        --severity_levels $SEVERITY \
        --if_${corr}
    echo "Done with $corr"
done

# 复制干净图像
echo "Copying clean images..."
python creat_corruption.py \
    --image_root $IMAGE_ROOT \
    --test_list $TEST_LIST \
    --folder_name leftImg8bit_sequence \
    --split test \
    --H $H \
    --W $W \
    --save_path $SAVE_PATH \
    --if_copy_clean

echo "All corruptions generated!"
```

### 3.4 输出目录结构

生成的Cityscapes-C目录结构：
```
data_cityscapes_corruptions/
├── clean/
│   ├── aachen/
│   │   ├── aachen_000000_leftImg8bit.png
│   │   └── ...
│   └── ...
├── brightness/
│   ├── severity_1/
│   │   ├── aachen/
│   │   │   ├── aachen_000000_leftImg8bit.png
│   │   │   └── ...
│   │   └── ...
│   ├── severity_2/
│   ├── severity_3/
│   ├── severity_4/
│   └── severity_5/
├── fog/
│   └── ...
└── ...
```

---

## 四、使用Cityscapes-C进行评估

### 4.1 修改评估脚本

在评估脚本中，将图像路径指向扰动后的图像：

```python
# 原始路径
image_path = os.path.join(data_path, 'leftImg8bit_sequence', 'test', city, frame_name + '_leftImg8bit.png')

# 扰动后路径（例如fog, severity_3）
corruption = 'fog'
severity = 3
image_path = os.path.join(data_path, 'data_cityscapes_corruptions', corruption, f'severity_{severity}', city, frame_name + '_leftImg8bit.png')
```

### 4.2 批量评估所有扰动

创建评估脚本 `eval_cityscapes_c.py`：

```python
import os
import subprocess

corruptions = [
    'brightness', 'dark', 'fog', 'frost', 'snow', 'contrast',
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
    'elastic', 'color_quant', 'gaussian_noise', 'impulse_noise',
    'shot_noise', 'iso_noise', 'pixelate', 'jpeg_compression'
]

severities = [1, 2, 3, 4, 5]

for corruption in corruptions:
    for severity in severities:
        print(f"Evaluating {corruption} severity {severity}...")
        # 修改数据路径指向扰动图像
        # 运行评估脚本
        subprocess.run([
            'python', 'evaluate_cityscapes_vectta.py',
            '--eval_data_path', f'data_cityscapes_corruptions/{corruption}/severity_{severity}',
            # ... 其他参数
        ])
```

---

## 五、完整示例命令

### 5.1 生成单个扰动类型

```bash
python creat_corruption.py \
    --image_root /data/cityscapes/leftImg8bit_sequence/test \
    --test_list SfMNeXt-Impl/splits/cityscapes/test_files.txt \
    --folder_name leftImg8bit_sequence \
    --split test \
    --H 320 \
    --W 1024 \
    --save_path data_cityscapes_corruptions \
    --severity_levels [1,2,3,4,5] \
    --if_fog \
    --seed 42
```

### 5.2 生成所有扰动（推荐）

```bash
# 使用批量脚本
chmod +x generate_cityscapes_c.sh
./generate_cityscapes_c.sh
```

### 5.3 只生成特定严重程度

```bash
# 只生成严重程度3的扰动
python creat_corruption.py \
    --image_root ... \
    --severity_levels [3] \
    --if_fog
```

---

## 六、注意事项

1. **存储空间**: Cityscapes-C会占用大量存储空间（每个扰动类型×5个严重程度×原始图像大小）

2. **生成时间**: 生成所有扰动可能需要较长时间，建议使用多进程或分批生成

3. **图像尺寸**: 确保 `--H` 和 `--W` 参数与评估时使用的尺寸一致

4. **文件路径**: 确保 `--image_root` 和 `--test_list` 路径正确

5. **严重程度**: 通常使用1-5的严重程度，数值越大扰动越强

6. **随机种子**: 使用 `--seed` 确保结果可复现

---

## 七、验证生成的扰动

生成后，可以检查：

```python
import os
from PIL import Image

corruption = 'fog'
severity = 3
image_path = 'data_cityscapes_corruptions/fog/severity_3/aachen/aachen_000000_leftImg8bit.png'

if os.path.exists(image_path):
    img = Image.open(image_path)
    print(f"Image size: {img.size}")
    print(f"Image mode: {img.mode}")
else:
    print("Image not found!")
```

---

## 八、参考

- Cityscapes-C参考ImageNet-C的设计
- 严重程度1-5对应不同的扰动强度
- 通常报告所有严重程度的平均性能（mCE: mean Corruption Error）






























