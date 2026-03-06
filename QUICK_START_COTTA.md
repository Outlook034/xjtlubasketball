# CoTTA 快速开始指南

## 单次评估命令

### 评估单个扰动

```bash
# 评估 clean 数据
python cotta_sqldepth.py \
    --data_path ../autodl-tmp/data/kitti_c/clean/1/kitti_data \
    --load_weights_folder ../autodl-tmp/KITTI_192x640_models \
    --eval_split eigen \
    --backbone resnet_lite \
    --height 192 \
    --width 640 \
    --batch_size 1 \
    --cotta_steps 1 \
    --lr 1e-5

# 或使用脚本
bash run_cotta_eval.sh clean 1
```

### 评估特定扰动

```bash
# 评估 fog 扰动，严重级别 3
python cotta_sqldepth.py \
    --data_path ../autodl-tmp/data/kitti_c/fog/3/kitti_data \
    --load_weights_folder ../autodl-tmp/KITTI_192x640_models \
    --eval_split eigen \
    --backbone resnet_lite \
    --height 192 \
    --width 640 \
    --batch_size 1 \
    --cotta_steps 1 \
    --lr 1e-5

# 或使用脚本
bash run_cotta_eval.sh fog 3
```

## 批量评估所有扰动

```bash
python batch_eval_cotta_kitti_c.py
```

## 对比原始模型性能（不使用 CoTTA）

```bash
python cotta_sqldepth.py \
    --data_path ../autodl-tmp/data/kitti_c/clean/1/kitti_data \
    --load_weights_folder ../autodl-tmp/KITTI_192x640_models \
    --eval_split eigen \
    --backbone resnet_lite \
    --height 192 \
    --width 640 \
    --batch_size 1 \
    --disable_cotta
```

## 常用参数说明

- `--cotta_steps`: CoTTA 适应步数（默认：1）
- `--lr`: 学习率（默认：1e-5，建议范围：1e-6 到 1e-4）
- `--mt_alpha`: EMA 更新率（默认：0.999）
- `--num_aug`: 数据增强次数（默认：32，可减少到 16 以加快速度）
- `--disable_cotta`: 禁用 CoTTA，直接使用原始模型

## 性能优化建议

1. **加快速度**：减少 `--num_aug` 到 16 或 8
2. **提高精度**：增加 `--cotta_steps` 到 2-3
3. **稳定训练**：降低 `--lr` 到 1e-6
4. **使用 episodic 模式**：添加 `--cotta_episodic`（每个样本重置模型）






