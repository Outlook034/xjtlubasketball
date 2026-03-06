# SQLdepth TTA 在 KITTI-C 上的评估指南

本指南说明如何使用 TTA（Test Time Adaptation）方法在 KITTI-C 数据集上评估 SQLdepth 模型。

## 文件说明

- `tta_sqldepth_kitti_c.py`: 主要的 TTA 评估脚本，支持 VECTTA 和 EATA 两种方法
- `batch_eval_sqldepth_kitti_c.py`: 批量评估脚本，用于在 KITTI-C 的所有扰动和严重级别上运行评估

## 使用方法

### 1. 生成 Clean 伪标签

首先需要在 clean 数据上生成伪标签，用于后续的 TTA 适应：

```bash
python tta_sqldepth_kitti_c.py \
    --mode generate_clean \
    --data_path /path/to/kitti/clean/data \
    --load_weights_folder /path/to/model/weights \
    --backbone resnet_lite \
    --height 192 \
    --width 640 \
    --batch_size 16 \
    --num_layers 50 \
    --num_features 256 \
    --model_dim 32 \
    --patch_size 16 \
    --dim_out 64 \
    --query_nums 64 \
    --min_depth 0.001 \
    --max_depth 80.0 \
    --clean_pred_path clean_pred_disps.npy
```

### 2. 单次 TTA 评估

在单个 KITTI-C 扰动上运行 TTA 评估：

```bash
python tta_sqldepth_kitti_c.py \
    --mode adapt \
    --tta_mode vec \
    --data_path /path/to/kitti_c/corruption/severity/kitti_data \
    --load_weights_folder /path/to/model/weights \
    --clean_pred_path clean_pred_disps.npy \
    --vec_steps 15 \
    --vec_update_mode bn_decoder \
    --vec_lr 1e-4 \
    --vec_early_stop \
    --vec_early_stop_patience 3 \
    --vec_grad_clip 1.0 \
    --backbone resnet_lite \
    --height 192 \
    --width 640 \
    --batch_size 16 \
    --num_layers 50 \
    --num_features 256 \
    --model_dim 32 \
    --patch_size 16 \
    --dim_out 64 \
    --query_nums 64 \
    --min_depth 0.001 \
    --max_depth 80.0
```

### 3. 批量评估所有 KITTI-C 扰动

使用批量评估脚本在 KITTI-C 的所有扰动和严重级别上运行评估：

```bash
python batch_eval_sqldepth_kitti_c.py
```

在运行前，请修改 `batch_eval_sqldepth_kitti_c.py` 中的以下参数：

- `corruption_root`: KITTI-C 数据集的根目录路径
- `--load_weights_folder`: 模型权重文件夹路径
- `--backbone`: 模型骨干网络类型
- 其他模型相关参数（根据你的模型配置调整）

## 参数说明

### TTA 模式

- `--tta_mode`: 选择 TTA 方法
  - `vec`: VECTTA（视角等变一致测试时适应）
  - `eata`: EATA（基于伪标签的测试时适应）

### VECTTA 参数

- `--vec_steps`: TTA 适应步数（默认: 5）
- `--vec_update_mode`: 参数更新模式
  - `bn_only`: 只更新 BatchNorm 层
  - `bn_decoder`: 更新 decoder 和所有 BN 层
  - `last_layers`: 更新最后几层和 BN 层
  - `all`: 更新整个模型
- `--vec_lr`: 学习率（默认: 1e-4）
- `--vec_early_stop`: 启用早停机制
- `--vec_early_stop_patience`: 早停耐心值（默认: 3）
- `--vec_grad_clip`: 梯度裁剪值（默认: 1.0）

### EATA 参数

- `--eata_steps`: EATA 适应步数（默认: 1）

### 模型参数

根据你的 SQLdepth 模型配置调整以下参数：

- `--backbone`: 骨干网络类型（如 `resnet_lite`, `resnet18_lite`, `eff_b5` 等）
- `--num_layers`: ResNet 层数（如 18, 34, 50, 101, 152）
- `--num_features`: 特征维度
- `--model_dim`: 模型维度
- `--patch_size`: Patch 大小
- `--dim_out`: 输出维度
- `--query_nums`: Query 数量

## 输出说明

评估完成后，脚本会输出以下指标：

- `abs_rel`: 绝对相对误差
- `sq_rel`: 平方相对误差
- `rmse`: 均方根误差
- `rmse_log`: 对数空间的均方根误差
- `a1`, `a2`, `a3`: 准确度指标（阈值分别为 1.25, 1.25², 1.25³）

批量评估脚本会生成一个日志文件，包含所有扰动的详细结果和平均结果。

## 注意事项

1. **Clean 伪标签**: 在运行 TTA 评估前，必须先生成 clean 伪标签。批量评估脚本会自动检查并生成。

2. **数据路径**: 确保 KITTI-C 数据集的目录结构正确：
   ```
   kitti_c/
   ├── corruption1/
   │   ├── 1/
   │   │   └── kitti_data/
   │   ├── 2/
   │   └── ...
   ├── corruption2/
   └── ...
   ```

3. **模型权重**: 确保模型权重文件路径正确，需要包含 `encoder.pth` 和 `depth.pth`。

4. **GPU 内存**: TTA 需要额外的 GPU 内存用于反向传播，如果遇到内存不足，可以减小 `--batch_size`。

5. **评估时间**: TTA 评估比标准评估慢，因为需要对每个样本进行适应。批量评估可能需要较长时间。

## 示例结果

评估完成后，你会看到类似以下的输出：

```
=== 各扰动平均结果 ===
dark            : 0.123 | 1.234 | 4.567 | 0.234 | 0.789 | 0.901 | 0.945
snow            : 0.145 | 1.456 | 5.123 | 0.256 | 0.756 | 0.889 | 0.934
...

=== 所有扰动平均 ===
0.134 | 1.345 | 4.890 | 0.245 | 0.773 | 0.895 | 0.940
```

日志文件会保存在当前目录，文件名格式为 `sqldepth_tta_evaluation_log_YYYYMMDD_HHMMSS.txt`。


