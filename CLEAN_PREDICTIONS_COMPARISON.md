# Cityscapes vs Kitti Clean伪标签处理对比

## 发现的不一致问题

### 1. 生成阶段（generate_clean）的差异

#### Cityscapes (`cityscapes_vectta.py`)
```python
# 生成clean伪标签时
if images.max() > 1.0:
    images = images / 255.0
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
images_norm = (images - mean) / std

output = depth_decoder(encoder(images_norm))
pred_disp = output[("disp", 0)]
```
**特点**：
- ✅ 显式进行图像归一化（除以255，然后标准化）
- ✅ 使用 `depth_decoder(encoder(images_norm))` 直接调用

#### Kitti (`tta_sqldepth_kitti_c.py`)
```python
# 生成clean伪标签时
images = batch[("color", 0, 0)].cuda()
pred_disp = model(images)  # 直接使用model
```
**特点**：
- ❌ **没有显式的图像归一化**
- ✅ 使用 `model(images)` 包装器

**问题**：如果Kitti数据集的图像预处理在DataLoader中完成，这可能没问题。但如果不在DataLoader中，可能导致不一致。

---

### 2. 使用阶段（adapt）的差异

#### Cityscapes (`cityscapes_vectta.py`)
```python
# 加载和使用clean伪标签
clean_disp_batch = clean_pred_disps[idx:idx+bs]
# 确保形状是 [B, H, W] 或 [B, 1, H, W]
if clean_disp_batch.ndim == 2:
    clean_disp_batch = clean_disp_batch[np.newaxis, ...]
if clean_disp_batch.ndim == 3:
    clean_disp_batch = clean_disp_batch[:, np.newaxis, ...]  # [B, H, W] -> [B, 1, H, W]
clean_depth = torch.from_numpy(clean_disp_batch).float().cuda()
```
**特点**：
- ✅ 显式处理形状转换：`[B, H, W]` → `[B, 1, H, W]`
- ✅ 处理了2D和3D的情况

#### Kitti (`tta_sqldepth_kitti_c.py`)
```python
# 加载和使用clean伪标签
clean_depth = torch.from_numpy(clean_pred_disps[idx:idx+images.shape[0]]).cuda()
```
**特点**：
- ❌ **没有显式的形状转换**
- ⚠️ 直接传递，依赖VecTTA内部处理

**问题**：虽然VecTTA内部会处理形状，但两个脚本的处理方式不一致。

---

### 3. 图像归一化的一致性

#### Cityscapes
```python
# 在adapt阶段也进行归一化
if images.max() > 1.0:
    images = images / 255.0
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
images_norm = (images - mean) / std
vectta(images_norm, K, clean_depth=clean_depth)
```

#### Kitti
```python
# 直接传递images，没有显式归一化
adapt_model(images, K, clean_depth=clean_depth)
```

**问题**：如果Kitti的DataLoader已经做了归一化，那没问题。但两个脚本的处理方式不一致。

---

## 总结：不一致的地方

### ❌ 主要不一致

1. **图像归一化**：
   - Cityscapes：显式归一化（generate_clean和adapt都归一化）
   - Kitti：可能依赖DataLoader，没有显式归一化

2. **clean伪标签形状处理**：
   - Cityscapes：显式转换 `[B, H, W]` → `[B, 1, H, W]`
   - Kitti：直接传递，依赖VecTTA内部处理

3. **模型调用方式**：
   - Cityscapes：`depth_decoder(encoder(images_norm))`
   - Kitti：`model(images)`（包装器）

### ✅ 一致的地方

1. 保存格式：都是 `[N, H, W]` 的npy文件
2. 加载方式：都使用 `np.load()`
3. 传递给VecTTA：都作为 `clean_depth` 参数
4. VecTTA内部处理：两个脚本使用的VecTTA类处理clean_depth的方式一致

---

## 建议修复

### 方案1：统一Kitti的处理方式（推荐）

修改 `tta_sqldepth_kitti_c.py` 使其与Cityscapes一致：

```python
# generate_clean阶段
if args.mode == 'generate_clean':
    model.eval()
    pred_disps = []
    with torch.no_grad():
        for batch in dataloader:
            images = batch[("color", 0, 0)].cuda()
            
            # 添加归一化（与Cityscapes一致）
            if images.max() > 1.0:
                images = images / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
            images_norm = (images - mean) / std
            
            pred_disp = model(images_norm)  # 使用归一化后的图像
            pred_disp = pred_disp.cpu().numpy()
            if pred_disp.ndim == 4:
                pred_disp = pred_disp[:, 0, :, :]
            pred_disps.append(pred_disp)
    # ... 保存

# adapt阶段
for batch in dataloader:
    images = batch[("color", 0, 0)].cuda()
    K = batch[("K", 0)].cuda()
    
    # 添加归一化
    if images.max() > 1.0:
        images = images / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
    images_norm = (images - mean) / std
    
    # 统一clean_depth的形状处理
    clean_depth = None
    if clean_pred_disps is not None:
        clean_disp_batch = clean_pred_disps[idx:idx+images.shape[0]]
        if clean_disp_batch.ndim == 2:
            clean_disp_batch = clean_disp_batch[np.newaxis, ...]
        if clean_disp_batch.ndim == 3:
            clean_disp_batch = clean_disp_batch[:, np.newaxis, ...]
        clean_depth = torch.from_numpy(clean_disp_batch).float().cuda()
        idx += images.shape[0]
    
    pred_disp = adapt_model(images_norm, K, clean_depth=clean_depth)
```

### 方案2：检查DataLoader的预处理

如果Kitti的DataLoader已经做了归一化，那么当前代码可能是正确的。需要检查：
- `datasets.kitti_dataset.KITTIRAWDataset` 的预处理
- 是否在DataLoader中进行了归一化

---

## 验证方法

### 1. 检查生成的clean伪标签
```python
# 对比两个数据集生成的clean伪标签
cityscapes_clean = np.load('clean_pred_disps_cityscapes.npy')
kitti_clean = np.load('clean_pred_disps.npy')

print(f"Cityscapes shape: {cityscapes_clean.shape}")
print(f"Kitti shape: {kitti_clean.shape}")
print(f"Cityscapes range: [{cityscapes_clean.min():.4f}, {cityscapes_clean.max():.4f}]")
print(f"Kitti range: [{kitti_clean.min():.4f}, {kitti_clean.max():.4f}]")
```

### 2. 检查图像预处理
```python
# 在generate_clean阶段打印图像统计
print(f"Images range: [{images.min():.2f}, {images.max():.2f}]")
print(f"Images mean: {images.mean():.4f}")
print(f"Images std: {images.std():.4f}")
```

### 3. 对比实验结果
- 使用相同的模型权重
- 在相同的数据集上生成clean伪标签
- 对比TTA效果差异

---

## 结论

**当前状态**：两个脚本的clean伪标签处理**不完全一致**，主要差异在：
1. 图像归一化的显式处理
2. clean_depth形状转换的显式处理

**建议**：统一两个脚本的处理方式，确保clean伪标签的生成和使用流程一致，这样可以：
- 提高代码可维护性
- 避免因处理不一致导致的性能差异
- 便于对比实验
