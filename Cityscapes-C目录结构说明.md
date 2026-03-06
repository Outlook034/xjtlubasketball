# Cityscapes-C 生成后的目录结构

## 一、完整目录结构

运行完 `create_cityscapes_corruptions.py` 脚本后，生成的目录结构如下：

```
data_cityscapes_corruptions/          # --save_path 指定的根目录
│
├── clean/                            # 干净图像（如果使用 --if_copy_clean）
│   ├── aachen/
│   │   ├── aachen_000000_leftImg8bit.png
│   │   ├── aachen_000001_leftImg8bit.png
│   │   └── ...
│   ├── bochum/
│   │   ├── bochum_000000_leftImg8bit.png
│   │   └── ...
│   └── ...
│
├── brightness/                       # 亮度扰动
│   ├── severity_1/
│   │   ├── aachen/
│   │   │   ├── aachen_000000_leftImg8bit.png
│   │   │   ├── aachen_000001_leftImg8bit.png
│   │   │   └── ...
│   │   ├── bochum/
│   │   │   └── ...
│   │   └── ...
│   ├── severity_2/
│   │   └── ...
│   ├── severity_3/
│   │   └── ...
│   ├── severity_4/
│   │   └── ...
│   └── severity_5/
│       └── ...
│
├── fog/                              # 雾化扰动
│   ├── severity_1/
│   │   └── ...
│   ├── severity_2/
│   ├── severity_3/
│   ├── severity_4/
│   └── severity_5/
│
├── dark/                             # 暗化扰动
│   └── ...
│
├── frost/                            # 霜冻扰动
│   └── ...
│
├── snow/                             # 降雪扰动
│   └── ...
│
├── contrast/                         # 对比度扰动
│   └── ...
│
├── defocus_blur/                     # 散焦模糊
│   └── ...
│
├── glass_blur/                       # 玻璃模糊
│   └── ...
│
├── motion_blur/                      # 运动模糊
│   └── ...
│
├── zoom_blur/                        # 缩放模糊
│   └── ...
│
├── elastic/                          # 弹性变形
│   └── ...
│
├── color_quant/                      # 颜色量化
│   └── ...
│
├── gaussian_noise/                   # 高斯噪声
│   └── ...
│
├── impulse_noise/                    # 脉冲噪声
│   └── ...
│
├── shot_noise/                       # 散粒噪声
│   └── ...
│
├── iso_noise/                        # ISO噪声
│   └── ...
│
├── pixelate/                         # 像素化
│   └── ...
│
└── jpeg_compression/                 # JPEG压缩
    └── ...
```

## 二、目录结构说明

### 2.1 根目录
- **名称**: `data_cityscapes_corruptions` (由 `--save_path` 参数指定)
- **位置**: 脚本运行时的当前工作目录下

### 2.2 扰动类型目录
每个扰动类型（如 `brightness`, `fog` 等）都会创建一个独立的目录。

### 2.3 严重程度子目录
每个扰动类型下，根据 `--severity_levels` 参数创建对应的严重程度目录：
- `severity_1/` - 严重程度1（最轻微）
- `severity_2/` - 严重程度2
- `severity_3/` - 严重程度3
- `severity_4/` - 严重程度4
- `severity_5/` - 严重程度5（最严重）

### 2.4 城市目录
在每个严重程度目录下，按照Cityscapes的原始结构，创建城市名称目录：
- `aachen/`
- `bochum/`
- `bremen/`
- `cologne/`
- `darmstadt/`
- `dusseldorf/`
- `erfurt/`
- `hamburg/`
- `hanover/`
- `jena/`
- `krefeld/`
- `leverkusen/`
- `monchengladbach/`
- `strasbourg/`
- `stuttgart/`
- `tubingen/`
- `ulm/`
- `weimar/`
- `zurich/`

### 2.5 图像文件
在每个城市目录下，保存扰动后的图像文件：
- 文件名格式: `{frame_name}_leftImg8bit.png`
- 例如: `aachen_000000_leftImg8bit.png`
- 图像尺寸: 由 `--H` 和 `--W` 参数指定（默认320×1024）

## 三、完整路径示例

### 3.1 干净图像路径
```
data_cityscapes_corruptions/clean/aachen/aachen_000000_leftImg8bit.png
```

### 3.2 扰动图像路径
```
data_cityscapes_corruptions/fog/severity_3/aachen/aachen_000000_leftImg8bit.png
data_cityscapes_corruptions/brightness/severity_5/bochum/bochum_000000_leftImg8bit.png
data_cityscapes_corruptions/gaussian_noise/severity_1/leverkusen/leverkusen_000039_000019_leftImg8bit.png
```

## 四、目录结构特点

### 4.1 保持原始结构
- 保留了Cityscapes原始的城市/帧名称目录结构
- 便于与原始数据集对应

### 4.2 分层组织
- 第一层: 扰动类型
- 第二层: 严重程度
- 第三层: 城市名称
- 第四层: 图像文件

### 4.3 易于访问
- 可以通过路径直接访问特定扰动类型和严重程度的图像
- 便于批量评估和对比

## 五、使用示例

### 5.1 访问特定扰动图像
```python
import os
from PIL import Image

# 访问fog扰动，严重程度3，aachen城市的第0帧
image_path = "data_cityscapes_corruptions/fog/severity_3/aachen/aachen_000000_leftImg8bit.png"
img = Image.open(image_path)
print(f"Image size: {img.size}")  # (1024, 320)
```

### 5.2 批量评估所有严重程度
```python
corruption = 'fog'
city = 'aachen'
frame_name = 'aachen_000000'

for severity in [1, 2, 3, 4, 5]:
    image_path = f"data_cityscapes_corruptions/{corruption}/severity_{severity}/{city}/{frame_name}_leftImg8bit.png"
    # 进行评估...
```

### 5.3 在评估脚本中使用
```python
# 修改评估脚本中的图像路径
corruption = 'fog'
severity = 3
image_path = os.path.join(
    'data_cityscapes_corruptions',
    corruption,
    f'severity_{severity}',
    city,
    frame_name + '_leftImg8bit.png'
)
```

## 六、目录大小估算

假设：
- 测试集图像数量: ~1500张
- 图像尺寸: 320×1024×3 (RGB)
- 每个PNG图像大小: ~500KB

**单个严重程度**: 1500 × 500KB ≈ 750MB
**单个扰动类型(5个严重程度)**: 750MB × 5 ≈ 3.75GB
**所有18种扰动类型**: 3.75GB × 18 ≈ 67.5GB
**加上clean**: 总计约 ~70GB

**注意**: 实际大小取决于图像压缩率和内容复杂度。

## 七、检查生成的目录

运行脚本后，可以使用以下命令检查：

```bash
# 查看根目录
ls data_cityscapes_corruptions/

# 查看某个扰动类型
ls data_cityscapes_corruptions/fog/

# 查看严重程度
ls data_cityscapes_corruptions/fog/severity_3/

# 查看城市目录
ls data_cityscapes_corruptions/fog/severity_3/aachen/

# 统计图像数量
find data_cityscapes_corruptions/fog/severity_3/ -name "*.png" | wc -l
```

## 八、注意事项

1. **目录结构一致性**: 所有扰动类型都遵循相同的目录结构
2. **文件名保持**: 图像文件名与原始Cityscapes保持一致
3. **路径分隔符**: Windows使用`\`，Linux/Mac使用`/`，Python会自动处理
4. **相对路径**: 建议使用相对路径或绝对路径，避免路径问题






























