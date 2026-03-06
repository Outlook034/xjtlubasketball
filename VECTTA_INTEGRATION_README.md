# VECTTA集成说明

## 概述

已成功将VECTTA（Viewpoint Equivariance Consistency Test-Time Adaptation）几何等变一致测试时适应算法集成到`evaluate_res50_depth_cityscapes_config.py`中。

## 主要功能

1. **几何等变一致性约束**: 通过注入虚拟微小相机位姿变换，构建深度预测的几何闭环一致性约束
2. **测试时适应**: 在推理过程中实时调整模型参数以提高性能
3. **支持两种模式**:
   - `vec`: VECTTA模式（默认）
   - `eata`: EATA模式

## 新增参数

### 基本参数
- `--use_vectta`: 启用VECTTA功能
- `--vectta_mode`: 选择模式（'vec' 或 'eata'）
- `--clean_pred_path`: clean伪标签文件路径

### VECTTA特定参数
- `--vec_steps`: 适应步数（默认5）
- `--vec_update_mode`: 参数更新模式（'bn_only', 'bn_decoder', 'last_layers', 'all'）
- `--vec_lr`: 学习率（默认1e-4）
- `--vec_early_stop`: 启用早停
- `--vec_early_stop_patience`: 早停耐心值（默认3）
- `--vec_grad_clip`: 梯度裁剪阈值（默认1.0）

### EATA特定参数
- `--eata_steps`: EATA适应步数（默认1）

## 使用方法

### 方式1：使用配置文件（推荐）

#### 1. 生成clean伪标签（第一步）
```bash
python evaluate_cityscapes_vectta.py @args_generate_clean.txt
```

#### 2. 使用VECTTA进行推理（第二步）
```bash
python evaluate_cityscapes_vectta.py @args_vectta_cityscapes_eval.txt
```

### 方式2：使用--mode参数

#### 1. 生成clean伪标签（第一步）
```bash
python evaluate_cityscapes_vectta.py @args_generate_clean_mode.txt
```

#### 2. 使用VECTTA进行推理（第二步）
```bash
python evaluate_cityscapes_vectta.py @args_vectta_mode.txt
```

### 3. 使用EATA模式
```bash
python evaluate_cityscapes_vectta.py @args_vectta_cityscapes_eval.txt --vectta_mode eata
```

### 4. 不使用VECTTA（原始模式）
```bash
python evaluate_cityscapes_vectta.py @args_vectta_cityscapes_eval.txt --no-use_vectta
```

### 5. 直接指定参数（不使用配置文件）
```bash
python evaluate_cityscapes_vectta.py \
    --eval_data_path data/CS_RAW/ \
    --eval_split cityscapes \
    --height 320 \
    --width 1024 \
    --model_dim 32 \
    --patch_size 16 \
    --dim_out 64 \
    --query_nums 64 \
    --min_depth 0.001 \
    --max_depth 80.0 \
    --load_weights_folder models/res50_cityscapes_320x1024/ \
    --eval_mono \
    --post_process \
    --save_pred_disps \
    --use_vectta \
    --vectta_mode vec \
    --clean_pred_path clean_pred_disps.npy \
    --vec_steps 5 \
    --vec_update_mode bn_decoder \
    --vec_lr 1e-4 \
    --vec_early_stop \
    --vec_early_stop_patience 3 \
    --vec_grad_clip 1.0
```

## 配置文件示例

已创建`args_vectta_cityscapes_eval.txt`配置文件，包含所有必要的VECTTA参数。

## 技术细节

1. **SQLDepthModel**: 将encoder和decoder组合为单一模型
2. **VECTTA类**: 实现几何等变一致性约束的核心算法
3. **EATA类**: 提供简单的测试时适应功能
4. **参数更新策略**: 支持多种参数更新模式，灵活控制适应范围

## 故障排除

### 常见错误

1. **参数冲突错误**：
   ```
   error: ambiguous option: --mode could match --model_name, --model_dim, --models_to_load
   ```
   **解决方案**：确保运行的是 `evaluate_cityscapes_vectta.py`，并且没有使用不存在的 `--mode` 参数。

2. **文件不存在错误**：
   ```
   FileNotFoundError: [Errno 2] No such file or directory: 'evaluate_cityscapes_vectta.py'
   ```
   **解决方案**：确保文件已重命名为 `evaluate_cityscapes_vectta.py`。

3. **模型权重文件不存在**：
   ```
   FileNotFoundError: [Errno 2] No such file or directory: 'models/res50_cityscapes_320x1024/encoder.pth'
   ```
   **解决方案**：确保模型权重文件存在于指定路径，或修改 `--load_weights_folder` 参数。

4. **Clean伪标签文件不存在**：
   ```
   Warning: Clean predictions file clean_pred_disps.npy not found. VECTTA will run without clean supervision.
   ```
   **解决方案**：这是警告信息，VECTTA仍会运行，但性能可能不如有clean监督时好。

### 正确的运行命令

```bash
# 基本运行（使用配置文件）
python evaluate_cityscapes_vectta.py @args_vectta_cityscapes_eval.txt

# 检查帮助信息
python evaluate_cityscapes_vectta.py --help

# 检查VECTTA相关参数
python evaluate_cityscapes_vectta.py --help | grep vectta
```

## 注意事项

1. 确保有足够的GPU内存进行测试时适应
2. 可以根据具体任务调整VECTTA参数以获得最佳性能
3. 如果遇到内存不足，可以减小batch size或减少VECTTA步数
4. 建议先在不使用VECTTA的情况下测试基本功能，然后再启用VECTTA
