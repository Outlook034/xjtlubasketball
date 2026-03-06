# SQLdepth + Ada-Depth 适应流程使用指南

本脚本将 ada-depth 的适应流程（Adapt 类）迁移到 SQLdepth 模型上，实现 SQLdepth 模型 + ada-depth 适应流程的组合。

## 核心特性

1. **SQLdepth 模型架构**：使用 SQLdepth 的 ResnetEncoderDecoder + Depth_Decoder_QueryTr 作为主模型
2. **Ada-Depth 适应流程**：完整保留 ada-depth 的适应机制：
   - EMA 模型用于生成稳定的伪标签
   - 自监督正则模型用于生成无监督伪标签
   - Pseudo label 生成和筛选机制
   - Scale-invariant loss 进行适应

## 文件说明

- `adaptation_sqldepth.py`: 主适应脚本，将 ada-depth 的 Adapt 类迁移到 SQLdepth

## 依赖要求

1. **ada-depth-main**：需要 ada-depth-main 目录可用，用于导入 DepthDecoder、ResnetEncoder 等正则模型组件
2. **SQLdepth 模型权重**：主模型的 encoder.pth 和 depth.pth
3. **正则模型权重**：用于自监督的正则模型权重（从 ada-depth 训练得到）

## 使用方法

### 1. 准备模型权重

- **主模型（SQLdepth）**：放在 `--load_weights_folder` 指定的目录
  - `encoder.pth`
  - `depth.pth`

- **正则模型（用于自监督）**：放在 `--reg_path` 指定的目录
  - `encoder.pth`
  - `depth.pth`
  - `pose_encoder.pth`
  - `pose.pth`

### 2. 运行适应

```bash
python adaptation_sqldepth.py \
    --data_path /path/to/kitti/data \
    --load_weights_folder /path/to/sqldepth/weights \
    --reg_path /path/to/regularization/model/weights \
    --model_name sqldepth_adapt \
    --log_dir ./logs \
    --dataset kitti \
    --eval_split eigen \
    --backbone resnet_lite \
    --height 192 \
    --width 640 \
    --batch_size 1 \
    --num_layers 50 \
    --num_features 256 \
    --model_dim 32 \
    --patch_size 16 \
    --dim_out 64 \
    --query_nums 64 \
    --min_depth 0.001 \
    --max_depth 80.0 \
    --scales 0 \
    --frame_ids 0 -1 1 \
    --thres 0.4 \
    --learning_rate 1e-5 \
    --weights_init pretrained
```

### 3. 在 KITTI-C 上测试

可以修改数据路径指向 KITTI-C 的某个扰动：

```bash
python adaptation_sqldepth.py \
    --data_path /path/to/kitti_c/corruption/severity/kitti_data \
    --load_weights_folder /path/to/sqldepth/weights \
    --reg_path /path/to/regularization/model/weights \
    ... (其他参数相同)
```

## 参数说明

### 模型参数

- `--backbone`: SQLdepth 骨干网络类型（resnet_lite, resnet18_lite, eff_b5 等）
- `--num_layers`: ResNet 层数（18, 34, 50, 101, 152）
- `--num_features`: 特征维度
- `--model_dim`: 模型维度
- `--patch_size`: Patch 大小
- `--dim_out`: 输出维度
- `--query_nums`: Query 数量

### 适应参数

- `--thres`: Pseudo label 筛选阈值（默认 0.4）
- `--learning_rate`: 适应学习率（默认 1e-5）
- `--reg_path`: 正则模型权重路径（必需）

### 数据参数

- `--data_path`: 数据路径
- `--eval_split`: 评估分割（eigen, benchmark 等）
- `--scales`: 多尺度列表（如 `0` 或 `0 1 2 3`）
- `--frame_ids`: 帧ID列表（如 `0 -1 1` 表示当前帧、前一帧、后一帧）

## 适应流程说明

1. **自监督模型更新**：首先更新正则模型（ResNet + DepthDecoder），使用自监督损失
2. **生成伪标签**：
   - 使用参考模型生成监督伪标签
   - 使用正则模型生成无监督伪标签
   - 使用 EMA 模型进行一致性检查
3. **主模型适应**：使用伪标签和 scale-invariant loss 更新 SQLdepth 模型
4. **EMA 更新**：更新 EMA 模型用于下一轮

## 输出说明

脚本会输出以下指标：

- **supervised w/o gt median scaling**: 不使用中位数缩放的监督指标（初始/教师/学生）
- **supervised w gt median scaling**: 使用中位数缩放的监督指标（初始/教师/学生）
- **self-supervised w gt median scaling**: 自监督指标（初始/适应后）

每个指标包括：abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3, median

## 注意事项

1. **ada-depth-main 依赖**：脚本需要从 ada-depth-main 导入 DepthDecoder 等组件，确保 ada-depth-main 目录在正确位置

2. **正则模型**：正则模型必须是从 ada-depth 训练得到的自监督模型，用于生成无监督伪标签

3. **数据格式**：数据需要包含 `color_uncrop`, `depth_gt_uncrop`, `K`, `inv_K` 等字段，与 ada-depth 的数据格式一致

4. **GPU 内存**：适应过程需要较多 GPU 内存，建议 batch_size=1

5. **评估模式**：脚本在适应过程中同时进行评估，每个 batch 都会计算误差指标

## 与原始 ada-depth 的区别

1. **主模型**：使用 SQLdepth 架构（ResnetEncoderDecoder + Depth_Decoder_QueryTr）替代 Swin Transformer + NewCRFDepth
2. **模型加载**：适配 SQLdepth 的权重加载方式
3. **输出处理**：SQLdepth decoder 返回字典格式，需要提取 `("disp", 0)`

## 故障排除

1. **ImportError: DepthDecoder**：
   - 确保 ada-depth-main 目录存在
   - 检查路径设置是否正确

2. **模型权重不匹配**：
   - 检查主模型权重是否与 SQLdepth 架构匹配
   - 检查正则模型权重是否完整

3. **数据格式错误**：
   - 确保数据加载器返回的字段与 ada-depth 一致
   - 检查 `color_uncrop`, `depth_gt_uncrop` 等字段是否存在


