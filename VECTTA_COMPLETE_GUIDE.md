# VECTTA完整使用指南

## 概述

本指南将帮助您完成VECTTA（几何等变一致测试时适应）的完整流程，包括生成clean伪标签和使用VECTTA进行测试时适应。

## 完整流程

### 步骤1：生成Clean伪标签

首先，您需要生成clean伪标签作为VECTTA的监督信号：

```bash
python evaluate_cityscapes_vectta.py @args_generate_clean.txt
```

这个命令会：
- 使用预训练模型在测试集上生成深度预测
- 将预测结果保存为 `clean_pred_disps.npy`
- 这些clean预测将作为VECTTA的监督信号

### 步骤2：使用VECTTA进行测试时适应

生成clean伪标签后，使用VECTTA进行测试时适应：

```bash
python evaluate_cityscapes_vectta.py @args_vectta_cityscapes_eval.txt
```

这个命令会：
- 加载clean伪标签
- 对每个测试样本进行VECTTA适应
- 输出适应后的深度预测结果
- 计算评估指标

## 配置文件说明

### args_generate_clean.txt
用于生成clean伪标签的配置：
```
--eval_data_path data/CS_RAW/          # 数据集路径
--eval_split cityscapes                # 评估分割
--height 320                           # 图像高度
--width 1024                           # 图像宽度
--model_dim 32                         # 模型维度
--patch_size 16                        # 补丁大小
--dim_out 64                           # 输出维度
--query_nums 64                        # 查询数量
--min_depth 0.001                      # 最小深度
--max_depth 80.0                       # 最大深度
--load_weights_folder models/res50_cityscapes_320x1024/  # 模型权重路径
--eval_mono                            # 单目评估
--post_process                         # 后处理
--generate_clean                       # 生成clean伪标签
--clean_pred_path clean_pred_disps.npy # clean伪标签保存路径
```

### args_vectta_cityscapes_eval.txt
用于VECTTA测试时适应的配置：
```
--eval_data_path data/CS_RAW/          # 数据集路径
--eval_split cityscapes                # 评估分割
--height 320                           # 图像高度
--width 1024                           # 图像宽度
--model_dim 32                         # 模型维度
--patch_size 16                        # 补丁大小
--dim_out 64                           # 输出维度
--query_nums 64                        # 查询数量
--min_depth 0.001                      # 最小深度
--max_depth 80.0                       # 最大深度
--load_weights_folder models/res50_cityscapes_320x1024/  # 模型权重路径
--eval_mono                            # 单目评估
--post_process                         # 后处理
--save_pred_disps                      # 保存预测结果
--use_vectta                           # 启用VECTTA
--vectta_mode vec                      # VECTTA模式
--clean_pred_path clean_pred_disps.npy # clean伪标签路径
--vec_steps 5                          # VECTTA步数
--vec_update_mode bn_decoder           # 参数更新模式
--vec_lr 1e-4                         # 学习率
--vec_early_stop                       # 早停
--vec_early_stop_patience 3            # 早停耐心值
--vec_grad_clip 1.0                    # 梯度裁剪
```

## 参数说明

### VECTTA核心参数

- `--use_vectta`: 启用VECTTA功能
- `--vectta_mode`: 选择模式（'vec' 或 'eata'）
- `--vec_steps`: VECTTA适应步数（默认5）
- `--vec_update_mode`: 参数更新模式
  - `bn_only`: 只更新BatchNorm层
  - `bn_decoder`: 更新decoder和BatchNorm层（推荐）
  - `last_layers`: 更新最后几层和BatchNorm层
  - `all`: 更新所有参数
- `--vec_lr`: VECTTA学习率（默认1e-4）
- `--vec_early_stop`: 启用早停机制
- `--vec_early_stop_patience`: 早停耐心值（默认3）
- `--vec_grad_clip`: 梯度裁剪阈值（默认1.0）

### Clean伪标签参数

- `--generate_clean`: 生成clean伪标签模式
- `--clean_pred_path`: clean伪标签保存/加载路径

## 运行示例

### 完整流程示例

```bash
# 1. 生成clean伪标签
echo "步骤1: 生成clean伪标签..."
python evaluate_cityscapes_vectta.py @args_generate_clean.txt

# 2. 使用VECTTA进行测试时适应
echo "步骤2: 使用VECTTA进行测试时适应..."
python evaluate_cityscapes_vectta.py @args_vectta_cityscapes_eval.txt

# 3. 查看结果
echo "步骤3: 查看结果..."
ls -la *.npy
```

### 不同模式示例

```bash
# 使用EATA模式
python evaluate_cityscapes_vectta.py @args_vectta_cityscapes_eval.txt --vectta_mode eata

# 不使用VECTTA（原始模式）
python evaluate_cityscapes_vectta.py @args_vectta_cityscapes_eval.txt --no-use_vectta

# 调整VECTTA参数
python evaluate_cityscapes_vectta.py @args_vectta_cityscapes_eval.txt --vec_steps 10 --vec_lr 2e-4
```

## 故障排除

### 常见问题

1. **模型权重文件不存在**
   ```
   FileNotFoundError: [Errno 2] No such file or directory: 'models/res50_cityscapes_320x1024/encoder.pth'
   ```
   **解决方案**: 确保模型权重文件存在于指定路径，或修改 `--load_weights_folder` 参数。

2. **数据集路径错误**
   ```
   FileNotFoundError: [Errno 2] No such file or directory: 'data/CS_RAW/'
   ```
   **解决方案**: 确保数据集路径正确，或修改 `--eval_data_path` 参数。

3. **内存不足**
   ```
   RuntimeError: CUDA out of memory
   ```
   **解决方案**: 减小batch size或减少VECTTA步数。

4. **Clean伪标签文件不存在**
   ```
   Warning: Clean predictions file clean_pred_disps.npy not found. VECTTA will run without clean supervision.
   ```
   **解决方案**: 先运行生成clean伪标签的命令。

## 输出文件

运行完成后，您会得到以下文件：

- `clean_pred_disps.npy`: Clean伪标签文件
- `disps_cityscapes_split.npy`: 预测的视差图
- `mono_cityscapes_errors.npy`: 评估误差
- `error_cityscapes_split.npy`: 误差图

## 性能优化建议

1. **GPU内存优化**: 如果遇到内存不足，可以：
   - 减小batch size
   - 减少VECTTA步数
   - 使用 `bn_only` 更新模式

2. **速度优化**: 如果希望加快速度，可以：
   - 减少VECTTA步数
   - 使用EATA模式而不是VECTTA模式
   - 禁用后处理

3. **精度优化**: 如果希望提高精度，可以：
   - 增加VECTTA步数
   - 调整学习率
   - 使用 `all` 更新模式

## 注意事项

1. 确保有足够的GPU内存进行测试时适应
2. 建议先在不使用VECTTA的情况下测试基本功能
3. 可以根据具体任务调整VECTTA参数以获得最佳性能
4. Clean伪标签的质量直接影响VECTTA的性能


