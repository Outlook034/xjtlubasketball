# VecTTA 完整流程详细分析

## 一、VecTTA 类结构概览

VecTTA (Viewpoint Equivariance Consistency Test-Time Adaptation) 是一个用于深度估计的测试时适应方法，通过几何一致性约束来优化模型参数。

### 核心组件
- **初始化方法**: `__init__()` - 设置基本参数和优化器
- **参数配置**: `configure_params_to_update()` - 选择可更新的参数
- **虚拟位姿采样**: `sample_virtual_poses()` - 生成虚拟相机位姿变换
- **深度回投影**: `back_project_depth()` - 将深度图变换到新视角
- **可见性掩码**: `create_visibility_mask()` - 创建有效区域掩码
- **损失计算**: 
  - `compute_scale_invariant_depth_error()` - 尺度不变深度误差
  - `compute_gradient_consistency_loss()` - 梯度一致性损失
- **主流程**: `forward()` - 完整的适应流程

---

## 二、方法详细分析

### 方法1: `__init__()` - 初始化

**位置**: 第42-54行

**功能**: 初始化VecTTA对象，设置基本参数和优化器

**输入参数**:
- `depth_model`: 深度估计模型（encoder + decoder）
- `update_mode`: 参数更新模式（'bn_only', 'bn_decoder', 'last_layers', 'all'）
- `optimizer_lr`: 优化器学习率（默认1e-4）
- `steps`: 适应步数（默认5）
- `early_stop`: 是否启用早停（默认True）
- `early_stop_patience`: 早停耐心值（默认3）
- `grad_clip`: 梯度裁剪阈值（默认1.0）

**执行步骤**:
1. 保存所有参数到实例变量
2. 调用 `configure_params_to_update()` 配置可更新参数
3. 创建 AdamW 优化器，只优化可更新参数

**关键代码**:
```python
self.configure_params_to_update()  # 配置哪些参数需要更新
self.optimizer = torch.optim.AdamW(self.params_to_update, lr=optimizer_lr)
```

---

### 方法2: `configure_params_to_update()` - 配置可更新参数

**位置**: 第56-110行

**功能**: 根据 `update_mode` 选择需要更新的模型参数，冻结其他参数

**执行逻辑**:

#### 步骤1: 冻结所有参数
```python
for p in self.depth_model.parameters():
    p.requires_grad = False
```

#### 步骤2: 根据模式解冻特定参数

**模式 'all'** (第61-65行):
- 解冻所有参数
- 适用于需要全面适应的情况

**模式 'bn_only'** (第67-76行):
- 只解冻所有 BatchNorm2d 层
- 设置 `track_running_stats = False`（测试时使用当前batch统计）
- 清空 `running_mean` 和 `running_var`
- **作用**: 只适应BN层的统计信息，计算量小，适合快速适应

**模式 'bn_decoder'** (第78-92行):
- 解冻整个decoder的所有参数
- 同时解冻所有BN层（包括encoder中的BN）
- **作用**: 适应解码器和归一化层，平衡效果和效率

**模式 'last_layers'** (第94-110行):
- 解冻名称包含'disp'的卷积层（通常是输出层）
- 同时解冻所有BN层
- **作用**: 只适应最后几层，最保守的更新策略

**输出**: 
- `self.params_to_update`: 可更新参数的列表

---

### 方法3: `sample_virtual_poses()` - 采样虚拟位姿

**位置**: 第112-132行

**功能**: 生成4个虚拟的微小相机位姿变换，用于构建几何一致性约束

**输入参数**:
- `batch_size`: batch大小
- `device`: 计算设备（GPU/CPU）
- `num_poses`: 位姿数量（默认4，固定）
- `xyz_range`: 平移范围（默认0.1米）
- `rpy_range`: 旋转范围（默认0.05弧度，约2.87度）

**位姿模式** (第115-120行):
```python
patterns = [
    [xyz_range, 0, 0, 0, rpy_range, 0],      # 沿X轴平移+绕X轴旋转
    [-xyz_range, 0, 0, 0, -rpy_range, 0],    # 沿X轴反向平移+反向旋转
    [0, 0, xyz_range, rpy_range, 0, 0],      # 沿Z轴平移+绕X轴旋转
    [0, xyz_range, 0, 0, 0, rpy_range],      # 沿Y轴平移+绕Z轴旋转
]
```

**执行步骤**:
1. 为每个模式创建平移向量 `translation` [B, 1, 1, 3]
2. 创建旋转向量 `axisangle` [B, 1, 1, 3]（轴角表示）
3. 返回位姿列表: `[(axisangle, translation), ...]`

**输出**: 
- `poses_list`: 包含4个位姿的列表，每个位姿是(axisangle, translation)元组

**设计原理**: 
- 使用微小变换（±0.1米，±0.05弧度）避免过大变形
- 4个不同方向的变换提供多角度的一致性约束

---

### 方法4: `back_project_depth()` - 深度回投影

**位置**: 第134-151行

**功能**: 将深度图从原视角变换到虚拟位姿对应的新视角

**输入参数**:
- `depth`: 原始深度图 [B, 1, H, W]
- `K`: 相机内参矩阵 [B, 3, 3] 或 [B, 3, 4]
- `pose`: 虚拟位姿 (axisangle, translation)

**执行步骤**:

#### 步骤1: 提取相机内参 (第137行)
```python
K_3x3 = K[:, :3, :3] if K.shape[-1] == 4 else K
```
- 如果K是4x4矩阵，只取前3x3部分

#### 步骤2: 初始化投影模块 (第138-139行)
```python
backproject = BackprojectDepth(batch_size, height, width).to(device)
project = Project3D(batch_size, height, width).to(device)
```
- `BackprojectDepth`: 将2D像素+深度反投影到3D点
- `Project3D`: 将3D点投影到2D像素坐标

#### 步骤3: 计算逆内参和投影矩阵 (第140-142行)
```python
inv_K = torch.inverse(K_3x3)  # 用于反投影
K_proj = torch.zeros(batch_size, 3, 4, device=device)
K_proj[:, :3, :3] = K_3x3  # 投影矩阵
```

#### 步骤4: 构建变换矩阵 (第143-144行)
```python
axisangle, translation = pose
T = transformation_from_parameters(axisangle[:, 0], translation[:, 0], invert=True)
```
- `invert=True`: 计算逆变换（从新视角看原视角）

#### 步骤5: 3D点反投影 (第145行)
```python
cam_points = backproject(depth, inv_K)
```
- 输入: 深度图 + 逆内参
- 输出: 3D点云 [B, 3, H*W]

#### 步骤6: 3D点变换并投影 (第146行)
```python
pix_coords = project(cam_points, K_proj, T)
```
- 将3D点通过变换矩阵T变换，然后投影到新视角
- 输出: 像素坐标 [B, H, W, 2]（归一化到[-1, 1]）

#### 步骤7: 提取Z坐标并采样 (第147-148行)
```python
z = cam_points[:, 2, :].reshape(batch_size, 1, height, width)
transformed_depth = F.grid_sample(z, pix_coords, mode='bilinear', ...)
```
- 提取3D点的Z坐标（深度值）
- 使用双线性插值采样到新视角的像素位置

#### 步骤8: 创建有效掩码 (第149-150行)
```python
valid_mask = ((pix_coords[:, :, :, 0] >= -1) & (pix_coords[:, :, :, 0] <= 1) &
              (pix_coords[:, :, :, 1] >= -1) & (pix_coords[:, :, :, 1] <= 1))
```
- 只保留投影在图像范围内的像素（归一化坐标在[-1,1]内）

#### 步骤9: 应用掩码 (第151行)
```python
return transformed_depth * valid_mask
```
- 将无效区域置为0

**输出**: 
- `transformed_depth`: 变换后的深度图 [B, 1, H, W]

**几何意义**: 
- 模拟从虚拟位姿观察同一场景，得到新视角的深度预测
- 如果模型预测准确，原视角深度和新视角回投影深度应该一致

---

### 方法5: `create_visibility_mask()` - 创建可见性掩码

**位置**: 第153-161行

**功能**: 创建有效像素的掩码，用于损失计算时过滤无效区域

**输入参数**:
- `depth`: 原始深度图 [B, 1, H, W]
- `back_depth`: 回投影深度图 [B, 1, H, W]
- `border_margin`: 边界边距（默认2像素）

**执行步骤**:

#### 步骤1: 检查数值有效性 (第154行)
```python
valid = torch.isfinite(back_depth) & (back_depth > 1e-6) & 
        torch.isfinite(depth) & (depth > 1e-6)
```
- `isfinite()`: 检查是否为有限值（非NaN、非Inf）
- `> 1e-6`: 过滤过小的深度值（避免数值不稳定）

#### 步骤2: 转换为浮点掩码 (第155行)
```python
mask = valid.float()
```

#### 步骤3: 去除边界区域 (第156-160行)
```python
if border_margin > 0:
    mask[:, :, :border_margin, :] = 0      # 上边界
    mask[:, :, -border_margin:, :] = 0      # 下边界
    mask[:, :, :, :border_margin] = 0      # 左边界
    mask[:, :, :, -border_margin:] = 0     # 右边界
```
- **原因**: 边界区域在回投影时容易出现无效值或畸变

**输出**: 
- `mask`: 有效像素掩码 [B, 1, H, W]，1表示有效，0表示无效

---

### 方法6: `compute_scale_invariant_depth_error()` - 尺度不变深度误差

**位置**: 第163-175行

**功能**: 计算尺度不变的深度误差，对全局尺度变化不敏感

**输入参数**:
- `depth`: 原始深度图 [B, 1, H, W]
- `back_depth`: 回投影深度图 [B, 1, H, W]
- `mask`: 有效像素掩码 [B, 1, H, W]

**执行步骤**:

#### 步骤1: 数值安全处理 (第164-166行)
```python
eps = 1e-6
depth_safe = torch.clamp(depth, min=eps)
back_depth_safe = torch.clamp(back_depth, min=eps)
```
- 确保所有值 > eps，避免log(0)或log(负数)

#### 步骤2: 默认掩码 (第167-168行)
```python
if mask is None:
    mask = torch.ones_like(depth_safe)
```

#### 步骤3: 检查有效像素 (第169-171行)
```python
valid = mask > 0.5
if valid.sum() == 0:
    return torch.zeros((), device=depth.device)  # 无有效像素，返回0损失
```

#### 步骤4: 计算对数差 (第172行)
```python
log_diff = (torch.log(depth_safe) - torch.log(back_depth_safe))[valid]
```
- 在log空间计算差异，等价于 log(depth/back_depth)
- 只计算有效像素

#### 步骤5: 尺度不变损失公式 (第173-174行)
```python
n = log_diff.numel()
loss = (log_diff.pow(2).sum() / n) - (log_diff.sum() ** 2) / (n ** 2)
```

**公式解析**:
- 第一项: `E[log²(diff)]` - 对数差的平方均值
- 第二项: `E²[log(diff)]` - 对数差均值的平方
- 最终: `Var(log(diff))` - 对数差的方差

**为什么尺度不变?**
- 如果两个深度图只差一个全局尺度因子s: `depth' = s * depth`
- 则 `log(depth') - log(depth) = log(s)` (常数)
- 方差为0，损失为0 ✓
- 因此对全局尺度变化不敏感

**输出**: 
- `loss`: 尺度不变深度误差（标量tensor）

---

### 方法7: `compute_gradient_consistency_loss()` - 梯度一致性损失

**位置**: 第177-191行

**功能**: 计算深度图在X和Y方向的梯度一致性，保持几何结构

**输入参数**:
- `depth`: 原始深度图 [B, 1, H, W]
- `back_depth`: 回投影深度图 [B, 1, H, W]
- `mask`: 有效像素掩码 [B, 1, H, W]

**执行步骤**:

#### 步骤1: 默认掩码 (第178-179行)
```python
if mask is None:
    mask = torch.ones_like(depth)
```

#### 步骤2: 定义梯度函数 (第180-183行)
```python
def gradient_x(img):
    return img[:, :, :, 1:] - img[:, :, :, :-1]  # 水平方向梯度

def gradient_y(img):
    return img[:, :, 1:, :] - img[:, :, :-1, :]  # 垂直方向梯度
```
- 使用前向差分计算梯度
- X方向: 右像素 - 左像素
- Y方向: 下像素 - 上像素

#### 步骤3: 计算两个深度图的梯度 (第184-185行)
```python
gx1, gy1 = gradient_x(depth), gradient_y(depth)      # 原深度图梯度
gx2, gy2 = gradient_x(back_depth), gradient_y(back_depth)  # 回投影深度图梯度
```

#### 步骤4: 计算梯度掩码 (第186-187行)
```python
mask_x = mask[:, :, :, 1:] * mask[:, :, :, :-1]  # X方向：相邻像素都有效
mask_y = mask[:, :, 1:, :] * mask[:, :, :-1, :]  # Y方向：相邻像素都有效
```
- 只有当相邻两个像素都有效时，梯度才有效

#### 步骤5: 计算梯度差异 (第188-189行)
```python
diff_x = torch.abs(gx1 - gx2) * mask_x  # X方向梯度差的绝对值
diff_y = torch.abs(gy1 - gy2) * mask_y  # Y方向梯度差的绝对值
```

#### 步骤6: 归一化损失 (第190-191行)
```python
denom = (mask_x.sum() + mask_y.sum()).clamp(min=1.0)
return (diff_x.sum() + diff_y.sum()) / denom
```
- 除以有效梯度数量，得到平均梯度差异

**作用**: 
- 确保深度图的局部几何结构（边缘、表面）在变换后保持一致
- 补充尺度不变损失，提供更细粒度的约束

**输出**: 
- `loss`: 梯度一致性损失（标量tensor）

---

### 方法8: `forward()` - 主适应流程

**位置**: 第193-259行

**功能**: VecTTA的核心方法，执行完整的测试时适应流程

**输入参数**:
- `image`: 输入图像 [B, 3, H, W]
- `K`: 相机内参 [B, 3, 3] 或 [B, 3, 4]
- `clean_depth`: Clean伪标签 [B, H, W] 或 [B, 1, H, W]（可选）

**完整执行流程**:

#### 阶段1: 初始化 (第194-198行)
```python
device = image.device
batch_size = image.shape[0]
self.depth_model.train()  # 切换到训练模式（启用BN更新）
best_loss = float('inf')
patience_counter = 0
```

#### 阶段2: 适应循环 (第200-252行)

**步骤2.1: 模型预测深度** (第201-204行)
```python
depth = self.depth_model(image)
if isinstance(depth, dict):
    depth = depth[("disp", 0)]  # 如果是字典，提取视差
```
- **关键**: 在训练模式下预测，保留梯度用于反向传播
- 输出: `depth` [B, 1, H, W]

**步骤2.2: 采样虚拟位姿** (第205行)
```python
poses = self.sample_virtual_poses(batch_size, device)
```
- 生成4个虚拟位姿变换

**步骤2.3: 计算几何一致性损失** (第206-214行)
```python
total_loss = depth.new_tensor(0.0)
for pose in poses:
    # 对每个虚拟位姿
    back_depth = self.back_project_depth(depth, K, pose)  # 回投影
    mask = self.create_visibility_mask(depth, back_depth)   # 创建掩码
    si_loss = self.compute_scale_invariant_depth_error(depth, back_depth, mask)
    grad_loss = self.compute_gradient_consistency_loss(depth, back_depth, mask)
    total_loss = total_loss + si_loss + 0.5 * grad_loss
total_loss = total_loss / max(len(poses), 1)  # 平均所有位姿的损失
```

**损失组成**:
- `si_loss`: 尺度不变深度误差
- `0.5 * grad_loss`: 梯度一致性损失（权重0.5）
- 对4个位姿的损失求平均

**步骤2.4: 计算Clean伪标签损失** (第216-236行)

这是**clean_depth使用的核心位置**！

```python
clean_loss = 0.0
if clean_depth is not None:  # 如果提供了clean伪标签
    # 步骤2.4.1: 维度调整
    if clean_depth.dim() == 3:
        clean_depth = clean_depth.unsqueeze(1)  # [B, H, W] → [B, 1, H, W]
    
    # 步骤2.4.2: 形状匹配
    if clean_depth.shape != depth.shape:
        # Batch维度匹配
        if clean_depth.shape[0] != depth.shape[0]:
            min_batch_size = min(clean_depth.shape[0], depth.shape[0])
            clean_depth = clean_depth[:min_batch_size]
            depth = depth[:min_batch_size]
        
        # 空间维度匹配（如果分辨率不同）
        if clean_depth.shape[2:] != depth.shape[2:]:
            clean_depth = F.interpolate(clean_depth, size=depth.shape[2:], 
                                      mode='bilinear', align_corners=False)
    
    # 步骤2.4.3: 计算L1损失
    if clean_depth.numel() > 0 and depth.numel() > 0:
        clean_loss = F.l1_loss(depth, clean_depth)  # L1损失
        total_loss = total_loss + 0.5 * clean_loss   # 权重0.5加入总损失
```

**详细说明**:
1. **维度检查**: 确保clean_depth是4D tensor
2. **Batch匹配**: 如果batch size不一致，取较小值
3. **空间匹配**: 如果分辨率不同，使用双线性插值调整clean_depth
4. **损失计算**: 使用L1损失（平均绝对误差）
5. **权重**: 0.5权重加入总损失，与几何损失平衡

**最终总损失公式**:
```
total_loss = (Σ(si_loss + 0.5*grad_loss) / 4) + 0.5*clean_loss
```

**步骤2.5: 反向传播和优化** (第238-242行)
```python
self.optimizer.zero_grad()  # 清零梯度
total_loss.backward()        # 反向传播
if self.grad_clip is not None and len(self.params_to_update) > 0:
    torch.nn.utils.clip_grad_norm_(self.params_to_update, self.grad_clip)  # 梯度裁剪
self.optimizer.step()       # 更新参数
```

**步骤2.6: 早停检查** (第244-252行)
```python
if self.early_stop:
    loss_val = float(total_loss.detach().item())
    if loss_val < best_loss:
        best_loss = loss_val
        patience_counter = 0  # 损失下降，重置计数器
    else:
        patience_counter += 1  # 损失未下降，增加计数器
    if patience_counter >= self.early_stop_patience:
        break  # 提前停止
```

#### 阶段3: 最终预测 (第254-259行)
```python
self.depth_model.eval()  # 切换回评估模式
with torch.no_grad():
    final_depth = self.depth_model(image)
    if isinstance(final_depth, dict):
        final_depth = final_depth[("disp", 0)]
return final_depth
```
- 在评估模式下进行最终预测（不使用BN的running stats）
- 无梯度计算，节省内存

**输出**: 
- `final_depth`: 适应后的深度预测 [B, 1, H, W]

---

## 三、完整流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                    VecTTA 完整执行流程                            │
└─────────────────────────────────────────────────────────────────┘

【初始化阶段】
┌─────────────────────────────────────────────────────────────────┐
│ 1. __init__(depth_model, update_mode, optimizer_lr, steps, ...) │
│    ├─ 保存参数                                                    │
│    ├─ configure_params_to_update()                               │
│    │   ├─ 冻结所有参数                                            │
│    │   ├─ 根据update_mode解冻特定参数:                            │
│    │   │   • 'bn_only': 只解冻BN层                                │
│    │   │   • 'bn_decoder': 解冻decoder + 所有BN层                 │
│    │   │   • 'last_layers': 解冻最后几层 + BN层                   │
│    │   │   • 'all': 解冻所有参数                                  │
│    │   └─ 返回可更新参数列表                                      │
│    └─ 创建AdamW优化器（只优化可更新参数）                         │
└─────────────────────────────────────────────────────────────────┘
                          ↓
【适应阶段 - forward(image, K, clean_depth=None)】
┌─────────────────────────────────────────────────────────────────┐
│ 2. 初始化适应环境                                                 │
│    ├─ device = image.device                                      │
│    ├─ batch_size = image.shape[0]                                │
│    ├─ depth_model.train()  # 切换到训练模式                       │
│    ├─ best_loss = inf                                            │
│    └─ patience_counter = 0                                       │
└─────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. 适应循环 (for step_idx in range(steps))                       │
│                                                                  │
│    ┌──────────────────────────────────────────────────────────┐ │
│    │ 3.1 模型预测深度                                            │ │
│    │     depth = depth_model(image)  # [B, 1, H, W]            │ │
│    │     # 保留梯度，用于反向传播                                │ │
│    └──────────────────────────────────────────────────────────┘ │
│                          ↓                                      │
│    ┌──────────────────────────────────────────────────────────┐ │
│    │ 3.2 采样虚拟位姿                                            │ │
│    │     poses = sample_virtual_poses(batch_size, device)      │ │
│    │     # 返回4个位姿: [(axisangle, translation), ...]        │ │
│    │     # 每个位姿包含微小平移(±0.1m)和旋转(±0.05rad)          │ │
│    └──────────────────────────────────────────────────────────┘ │
│                          ↓                                      │
│    ┌──────────────────────────────────────────────────────────┐ │
│    │ 3.3 计算几何一致性损失 (对每个位姿)                         │ │
│    │     total_loss = 0.0                                      │ │
│    │     for pose in poses:  # 4个位姿循环                      │ │
│    │       ┌────────────────────────────────────────────────┐ │ │
│    │       │ 3.3.1 深度回投影                                  │ │ │
│    │       │   back_depth = back_project_depth(depth, K, pose)│ │ │
│    │       │   ├─ 提取相机内参K_3x3                             │ │ │
│    │       │   ├─ 初始化BackprojectDepth和Project3D模块        │ │ │
│    │       │   ├─ 计算逆内参inv_K                               │ │ │
│    │       │   ├─ 构建变换矩阵T (从pose)                        │ │ │
│    │       │   ├─ cam_points = backproject(depth, inv_K)        │ │ │
│    │       │   │   # 2D像素+深度 → 3D点云                        │ │ │
│    │       │   ├─ pix_coords = project(cam_points, K, T)       │ │ │
│    │       │   │   # 3D点 → 新视角2D像素坐标                    │ │ │
│    │       │   ├─ z = cam_points[:, 2, :]                       │ │ │
│    │       │   ├─ transformed_depth = grid_sample(z, pix_coords)│ │ │
│    │       │   │   # 双线性插值采样到新视角                      │ │ │
│    │       │   ├─ valid_mask = 检查像素坐标是否在[-1,1]内       │ │ │
│    │       │   └─ return transformed_depth * valid_mask       │ │ │
│    │       └────────────────────────────────────────────────┘ │ │
│    │                          ↓                                │ │
│    │       ┌────────────────────────────────────────────────┐ │ │
│    │       │ 3.3.2 创建可见性掩码                              │ │ │
│    │       │   mask = create_visibility_mask(depth, back_depth)│ │ │
│    │       │   ├─ valid = isfinite & > 1e-6                    │ │ │
│    │       │   ├─ mask = valid.float()                         │ │ │
│    │       │   └─ 去除边界2像素 (border_margin=2)              │ │ │
│    │       └────────────────────────────────────────────────┘ │ │
│    │                          ↓                                │ │
│    │       ┌────────────────────────────────────────────────┐ │ │
│    │       │ 3.3.3 计算尺度不变深度误差                         │ │ │
│    │       │   si_loss = compute_scale_invariant_depth_error() │ │ │
│    │       │   ├─ depth_safe = clamp(depth, min=eps)          │ │ │
│    │       │   ├─ back_depth_safe = clamp(back_depth, min=eps)│ │ │
│    │       │   ├─ log_diff = log(depth_safe) - log(back_depth)│ │ │
│    │       │   └─ loss = Var(log_diff)                         │ │ │
│    │       │       # 方差公式: E[log²] - E²[log]               │ │ │
│    │       └────────────────────────────────────────────────┘ │ │
│    │                          ↓                                │ │
│    │       ┌────────────────────────────────────────────────┐ │ │
│    │       │ 3.3.4 计算梯度一致性损失                          │ │ │
│    │       │   grad_loss = compute_gradient_consistency_loss()│ │ │
│    │       │   ├─ gx1, gy1 = gradient_x/y(depth)             │ │ │
│    │       │   ├─ gx2, gy2 = gradient_x/y(back_depth)        │ │ │
│    │       │   ├─ mask_x/y = 相邻像素掩码                       │ │ │
│    │       │   ├─ diff_x/y = |gx1-gx2| * mask_x/y             │ │ │
│    │       │   └─ loss = (diff_x.sum + diff_y.sum) / denom   │ │ │
│    │       └────────────────────────────────────────────────┘ │ │
│    │                          ↓                                │ │
│    │       total_loss += si_loss + 0.5 * grad_loss            │ │
│    │     total_loss = total_loss / 4  # 平均4个位姿的损失      │ │
│    └──────────────────────────────────────────────────────────┘ │
│                          ↓                                      │
│    ┌──────────────────────────────────────────────────────────┐ │
│    │ 3.4 【Clean伪标签使用】                                     │ │
│    │     clean_loss = 0.0                                      │ │
│    │     if clean_depth is not None:                           │ │
│    │       ├─ 3.4.1 维度调整                                    │ │
│    │       │   if clean_depth.dim() == 3:                      │ │
│    │       │     clean_depth = clean_depth.unsqueeze(1)         │ │
│    │       │     # [B, H, W] → [B, 1, H, W]                    │ │
│    │       │                                                     │ │
│    │       ├─ 3.4.2 Batch维度匹配                               │ │
│    │       │   if clean_depth.shape[0] != depth.shape[0]:      │ │
│    │       │     min_batch = min(...)                          │ │
│    │       │     clean_depth = clean_depth[:min_batch]         │ │
│    │       │     depth = depth[:min_batch]                      │ │
│    │       │                                                     │ │
│    │       ├─ 3.4.3 空间维度匹配（分辨率调整）                    │ │
│    │       │   if clean_depth.shape[2:] != depth.shape[2:]:    │ │
│    │       │     clean_depth = F.interpolate(                   │ │
│    │       │       clean_depth, size=depth.shape[2:],            │ │
│    │       │       mode='bilinear', align_corners=False          │ │
│    │       │     )                                              │ │
│    │       │                                                     │ │
│    │       └─ 3.4.4 计算Clean损失                               │ │
│    │           if clean_depth.numel() > 0:                     │ │
│    │             clean_loss = F.l1_loss(depth, clean_depth)    │ │
│    │             total_loss = total_loss + 0.5 * clean_loss     │ │
│    │             # 权重0.5，与几何损失平衡                      │ │
│    └──────────────────────────────────────────────────────────┘ │
│                          ↓                                      │
│    ┌──────────────────────────────────────────────────────────┐ │
│    │ 3.5 反向传播和优化                                          │ │
│    │     optimizer.zero_grad()  # 清零梯度                      │ │
│    │     total_loss.backward()   # 反向传播                     │ │
│    │     clip_grad_norm_(params_to_update, grad_clip)          │ │
│    │     optimizer.step()        # 更新参数                     │ │
│    └──────────────────────────────────────────────────────────┘ │
│                          ↓                                      │
│    ┌──────────────────────────────────────────────────────────┐ │
│    │ 3.6 早停检查                                                │ │
│    │     if early_stop:                                        │ │
│    │       loss_val = total_loss.item()                        │ │
│    │       if loss_val < best_loss:                            │ │
│    │         best_loss = loss_val                              │ │
│    │         patience_counter = 0                              │ │
│    │       else:                                               │ │
│    │         patience_counter += 1                             │ │
│    │       if patience_counter >= early_stop_patience:         │ │
│    │         break  # 提前停止                                  │ │
│    └──────────────────────────────────────────────────────────┘ │
│                                                                  │
│    循环继续，直到完成所有steps或触发早停                          │
└─────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. 最终预测                                                      │
│    depth_model.eval()  # 切换回评估模式                          │
│    with torch.no_grad():                                        │
│      final_depth = depth_model(image)                           │
│    return final_depth  # [B, 1, H, W]                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## 四、Clean伪标签的完整生命周期

### 阶段1: 生成Clean伪标签（预处理）

```python
# 模式: generate_clean
model.eval()
pred_disps = []
with torch.no_grad():
    for batch in dataloader:
        images = batch[("color", 0, 0)].cuda()
        pred_disp = model(images)  # 原始模型预测
        pred_disps.append(pred_disp.cpu().numpy())
pred_disps = np.concatenate(pred_disps, axis=0)
np.save('clean_pred_disps.npy', pred_disps)  # 保存
```

### 阶段2: 加载Clean伪标签（适应前）

```python
# 模式: adapt
clean_pred_disps = np.load('clean_pred_disps.npy')  # 加载
idx = 0
```

### 阶段3: 传入VecTTA（每个batch）

```python
for batch in dataloader:
    images = batch[("color", 0, 0)].cuda()
    K = batch[("K", 0)].cuda()
    
    # 从数组中切片当前batch的clean伪标签
    clean_depth = torch.from_numpy(
        clean_pred_disps[idx:idx+images.shape[0]]
    ).cuda()
    idx += images.shape[0]
    
    # 传入VecTTA
    pred_disp = adapt_model(images, K, clean_depth=clean_depth)
```

### 阶段4: VecTTA内部使用（forward方法）

在 `forward()` 方法的第216-236行，clean_depth被使用：

1. **检查存在性**: `if clean_depth is not None:`
2. **维度调整**: 3D → 4D
3. **形状匹配**: Batch和空间维度对齐
4. **损失计算**: `clean_loss = F.l1_loss(depth, clean_depth)`
5. **加入总损失**: `total_loss += 0.5 * clean_loss`

---

## 五、关键设计要点

### 1. 为什么使用虚拟位姿？
- **几何先验**: 深度预测应该满足几何一致性
- **无监督**: 不需要真实标签，只利用几何约束
- **多角度约束**: 4个不同方向的变换提供丰富的约束

### 2. 为什么使用尺度不变损失？
- **深度尺度模糊**: 单目深度估计存在尺度不确定性
- **鲁棒性**: 对全局尺度变化不敏感，只关注相对深度关系

### 3. 为什么需要梯度一致性损失？
- **局部结构**: 保持深度图的边缘和表面结构
- **细粒度约束**: 补充尺度不变损失的不足

### 4. Clean伪标签的作用？
- **稳定参考**: 提供原始模型的预测作为参考
- **防止过拟合**: 避免适应过程偏离太远
- **监督信号**: 在无监督几何损失基础上增加有监督信号

### 5. 参数更新模式的选择？
- **bn_only**: 最快，只适应统计信息
- **bn_decoder**: 平衡效果和效率
- **last_layers**: 最保守，只适应输出层
- **all**: 最全面，但可能过拟合

---

## 六、损失函数总结

### 总损失公式

```
total_loss = 几何一致性损失 + Clean伪标签损失

其中:
几何一致性损失 = (Σ(尺度不变损失 + 0.5×梯度一致性损失) / 4)
Clean伪标签损失 = 0.5 × L1损失(depth, clean_depth)
```

### 各项损失的作用

| 损失项 | 权重 | 作用 | 特点 |
|--------|------|------|------|
| 尺度不变深度误差 | 1.0 | 约束深度值的尺度不变一致性 | 对全局尺度不敏感 |
| 梯度一致性损失 | 0.5 | 约束深度图的局部几何结构 | 保持边缘和表面 |
| Clean伪标签损失 | 0.5 | 约束与原始模型预测的一致性 | 防止过拟合 |

---

## 七、总结

VecTTA通过以下机制实现测试时适应：

1. **几何一致性约束**: 使用虚拟位姿变换构建闭环约束
2. **多损失组合**: 尺度不变损失 + 梯度一致性损失 + Clean伪标签损失
3. **选择性更新**: 只更新部分参数（BN层、decoder等）
4. **早停机制**: 防止过拟合
5. **Clean伪标签**: 提供稳定参考，平衡适应效果

整个流程在测试时对每个样本进行快速适应（通常5步），无需重新训练模型，即可提升深度估计的准确性。

