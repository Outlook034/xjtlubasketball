# Clean伪标签（Clean Predictions）使用指南

## 1. generate_clean 语法说明

### 基本语法
```bash
python tta_sqldepth_kitti_c.py --mode generate_clean [其他参数]
```

### 参数说明
- `--mode generate_clean`: 指定运行模式为"生成clean伪标签"
- `--clean_pred_path`: 指定保存路径（默认：`clean_pred_disps.npy`）
- 其他模型参数：需要与后续adapt模式使用相同的参数

### 完整示例
```bash
# 生成clean伪标签
python tta_sqldepth_kitti_c.py \
    --mode generate_clean \
    --data_path /path/to/kitti \
    --load_weights_folder /path/to/weights \
    --eval_split eigen \
    --backbone resnet_lite \
    --height 192 \
    --width 640 \
    --batch_size 16 \
    --clean_pred_path clean_pred_disps.npy
```

## 2. generate_clean 的工作原理

### 代码流程
```python
if args.mode == 'generate_clean':
    model.eval()  # 设置为评估模式
    pred_disps = []
    with torch.no_grad():  # 禁用梯度计算
        for batch in dataloader:
            images = batch[("color", 0, 0)].cuda()
            pred_disp = model(images)  # 使用原始模型预测（无TTA）
            pred_disp = pred_disp.cpu().numpy()
            if pred_disp.ndim == 4:
                pred_disp = pred_disp[:, 0, :, :]
            pred_disps.append(pred_disp)
    pred_disps = np.concatenate(pred_disps, axis=0)
    np.save(args.clean_pred_path, pred_disps)  # 保存为npy文件
```

### 关键特点
1. **不使用TTA**：直接使用原始模型进行预测
2. **禁用梯度**：`torch.no_grad()`确保不计算梯度，节省内存
3. **保存视差**：保存的是视差（disparity），不是深度（depth）
4. **形状**：最终保存的npy文件形状为 `[N, H, W]`，其中N是样本数量

## 3. Clean伪标签在TTA中的作用

### 在adapt模式中的使用
```python
# 加载clean伪标签
clean_pred_disps = np.load(args.clean_pred_path)

# 在TTA适应过程中使用
clean_loss = F.l1_loss(depth, clean_depth)
total_loss = total_loss + 0.5 * clean_loss  # 权重0.5
```

### 损失函数组成
VecTTA的总损失由三部分组成：
1. **几何一致性损失**（si_loss）：尺度不变深度误差
2. **梯度一致性损失**（grad_loss）：梯度一致性约束
3. **Clean伪标签损失**（clean_loss）：与clean预测的L1损失（权重0.5）

```
total_loss = (si_loss + 0.5 * grad_loss) + 0.5 * clean_loss
```

## 4. 为什么不同的npy文件效果不同？

### 影响因素

#### 4.1 模型状态差异
- **不同的checkpoint**：使用不同训练阶段的模型生成的clean伪标签质量不同
  - 训练充分的模型 → 更好的clean伪标签 → 更好的TTA效果
  - 欠训练的模型 → 较差的clean伪标签 → 可能误导TTA适应

#### 4.2 数据集匹配
- **训练数据集**：clean伪标签应该在与测试数据相似的数据集上生成
  - 如果clean伪标签在Kitti训练集上生成，用于Kitti测试集 → 效果好
  - 如果clean伪标签在其他数据集生成，用于Kitti测试集 → 效果可能差

#### 4.3 数据预处理一致性
- **图像预处理**：生成clean伪标签时的预处理必须与adapt模式一致
  - 图像尺寸（height, width）
  - 归一化方式
  - 数据增强（如果有）

#### 4.4 模型架构匹配
- **backbone类型**：必须使用相同的backbone生成clean伪标签
  - `resnet_lite` vs `resnet` vs `eff_b5` → 不同的特征提取能力
  - 不同的backbone生成的clean伪标签质量差异很大

#### 4.5 视差vs深度
- **单位一致性**：确保clean伪标签的单位与模型输出一致
  - 如果模型输出视差，clean伪标签也应该是视差
  - 如果模型输出深度，需要转换

### 质量评估指标

#### 好的clean伪标签特征：
1. **高精度**：在测试集上的误差较小
2. **一致性**：与模型在相同数据上的预测一致
3. **稳定性**：不同batch之间预测稳定

#### 差的clean伪标签特征：
1. **系统性偏差**：整体预测偏大或偏小
2. **噪声多**：预测结果不稳定
3. **与测试数据不匹配**：来自不同域的数据

## 5. 最佳实践

### 5.1 生成clean伪标签的建议
```bash
# 1. 使用与测试数据相同的数据集
# 2. 使用训练好的模型（不是中间checkpoint）
# 3. 确保所有参数与adapt模式一致

python tta_sqldepth_kitti_c.py \
    --mode generate_clean \
    --data_path /path/to/kitti \
    --load_weights_folder /path/to/trained_weights \
    --eval_split eigen \
    --backbone resnet_lite \
    --height 192 \
    --width 640 \
    --model_dim 32 \
    --patch_size 16 \
    --dim_out 64 \
    --query_nums 64 \
    --clean_pred_path clean_pred_disps_eigen.npy
```

### 5.2 验证clean伪标签质量
```python
# 可以单独评估clean伪标签的质量
import numpy as np
from utils import compute_errors

clean_preds = np.load('clean_pred_disps.npy')
gt_depths = np.load('gt_depths.npz')['data']

errors = []
for i in range(len(clean_preds)):
    errors.append(compute_errors(gt_depths[i], clean_preds[i]))

mean_errors = np.array(errors).mean(0)
print("Clean伪标签质量:", mean_errors)
```

### 5.3 对比实验
```bash
# 实验1：使用高质量clean伪标签
python tta_sqldepth_kitti_c.py --mode adapt \
    --clean_pred_path clean_pred_disps_high_quality.npy

# 实验2：使用低质量clean伪标签
python tta_sqldepth_kitti_c.py --mode adapt \
    --clean_pred_path clean_pred_disps_low_quality.npy

# 对比结果差异
```

## 6. 常见问题

### Q1: 可以不使用clean伪标签吗？
**A**: 可以。如果不提供`--clean_pred_path`或文件不存在，`clean_loss = 0`，TTA只使用几何一致性损失。

### Q2: clean伪标签必须与测试数据完全匹配吗？
**A**: 最好匹配。如果clean伪标签来自不同的数据集或不同的split，效果会下降。

### Q3: 可以使用ground truth作为clean伪标签吗？
**A**: 理论上可以，但这不是clean伪标签的初衷。clean伪标签应该是模型在clean数据上的预测，而不是GT。

### Q4: 如何选择clean伪标签的权重？
**A**: 当前代码中权重固定为0.5。如果需要调整，可以修改代码中的权重值：
```python
total_loss = total_loss + 0.5 * clean_loss  # 修改0.5为其他值
```

## 7. 总结

**generate_clean的作用**：
- 生成模型在clean数据上的预测作为伪标签
- 为TTA适应提供额外的监督信号

**为什么不同npy文件效果不同**：
1. 模型质量差异（训练充分程度）
2. 数据集匹配度（训练集vs测试集）
3. 预处理一致性（尺寸、归一化等）
4. 模型架构匹配（backbone类型）

**建议**：
- 使用训练好的模型生成clean伪标签
- 确保生成和使用的参数一致
- 在相同数据集上生成和使用clean伪标签
- 定期验证clean伪标签的质量
