import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from networks.pose_cnn import PoseCNN  # 导入PoseCNN模型
# 设置随机种子确保结果可重复
torch.manual_seed(42)
np.random.seed(42)

# 设置计算设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 配置参数
batch_size = 4
height = 192
width = 640
num_input_frames = 2  # 输入帧数（例如相邻帧）

# 创建模拟输入数据 - 两帧连接的图像
input_tensor = torch.rand(batch_size, 3 * num_input_frames, height, width).to(device)
print(f"模型输入维度: {input_tensor.shape}")

# 创建PoseCNN模型 (pose_set)
pose_set = PoseCNN(num_input_frames=num_input_frames).to(device)

# 打印模型结构
print(f"\nPoseCNN模型结构:")
print(pose_set)

# 执行前向传播 - PoseCNN
print("\n执行PoseCNN前向传播...")
with torch.no_grad():
    axisangle, translation = pose_set(input_tensor)

# 打印PoseCNN输出维度和值
print("\nPoseCNN输出维度:")
print(f"旋转轴角向量: {axisangle.shape}")
print(f"平移向量: {translation.shape}")

print("\nPoseCNN输出样本值:")
print(f"旋转轴角向量:")
print(axisangle.cpu().numpy())
print(f"平移向量:")
print(translation.cpu().numpy())

print("\n完整模型测试完成！")

# 加载pose.pth权重（包含整个PoseCNN模型）
pose_path = "..autodl-tmp/KITTI_192x640_models/pose.pth"

print(f"\n正在加载pose.pth权重: {pose_path}")

try:
    # 加载权重
    state_dict = torch.load(pose_path, map_location=device)
    print(f"成功加载权重文件，包含 {len(state_dict)} 个参数")
    
    # 打印权重参数名称
    print("\n权重参数名称:")
    for key in state_dict.keys():
        print(f"  {key}: {state_dict[key].shape}")
    
    # 尝试加载权重到模型
    missing_keys, unexpected_keys = pose_set.load_state_dict(state_dict, strict=False)
    
    print(f"\n加载结果:")
    print(f"缺失的键: {missing_keys}")
    print(f"多余的键: {unexpected_keys}")
    
    if not missing_keys and not unexpected_keys:
        print("✅ pose.pth包含完整的PoseCNN模型权重")
    elif missing_keys:
        print("❌ pose.pth缺少部分权重参数")
    elif unexpected_keys:
        print("⚠️ pose.pth包含多余的权重参数")
        
except Exception as e:
    print(f"❌ 加载权重失败: {e}")

# 设置为评估模式
pose_set.eval()

# 测试加载权重后的模型
print("\n测试加载权重后的模型...")
with torch.no_grad():
    axisangle_loaded, translation_loaded = pose_set(input_tensor)

print("加载权重后的输出:")
print(f"旋转轴角向量: {axisangle_loaded.shape}")
print(f"平移向量: {translation_loaded.shape}")
print(f"旋转轴角样本值: {axisangle_loaded.cpu().numpy()}")
print(f"平移样本值: {translation_loaded.cpu().numpy()}")

print("\n权重加载测试完成！")

# 使用真实图像测试（和原脚本一样的方法）
import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# 设置变换操作
transform = transforms.Compose([
    transforms.Resize((192, 640)),  # 调整为模型所需尺寸
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 读取两帧图像
frame1_path = "../autodl-tmp/data/kitti_c/gaussian_noise/2/kitti_data/2011_09_26/2011_09_26_drive_0002_sync/image_02/data/0000000006.png"
frame2_path = "../autodl-tmp/data/kitti_c/gaussian_noise/2/kitti_data/2011_09_26/2011_09_26_drive_0002_sync/image_02/data/0000000009.png"

frame1 = Image.open(frame1_path).convert('RGB')
frame2 = Image.open(frame2_path).convert('RGB')

# 预处理图像
frame1_tensor = transform(frame1).unsqueeze(0).to(device)
frame2_tensor = transform(frame2).unsqueeze(0).to(device)

# 将两帧图像连接起来作为输入
input_tensor = torch.cat([frame1_tensor, frame2_tensor], dim=1)
    
# PoseCNN前向传播
with torch.no_grad():
    poses = pose_set(input_tensor)
        
# 提取旋转和平移信息
rotation_tensor = poses[0]  # 形状为 [1, 2, 1, 3]
translation_tensor = poses[1]  # 形状为 [1, 2, 1, 3]

# 解析两帧之间的位姿
# 提取第一组位姿
rotation1 = rotation_tensor[0, 0, 0].cpu().numpy()
translation1 = translation_tensor[0, 0, 0].cpu().numpy()

# 提取第二组位姿
rotation2 = rotation_tensor[0, 1, 0].cpu().numpy()
translation2 = translation_tensor[0, 1, 0].cpu().numpy()

print("第一帧旋转参数:", rotation1)
print("第一帧平移参数:", translation1)
print("第二帧旋转参数:", rotation2)
print("第二帧平移参数:", translation2)

# 计算相对变换（和原脚本一样的方法）
def axis_angle_to_rotation_matrix(axis_angle):
    """将轴角表示转换为旋转矩阵"""
    angle = np.linalg.norm(axis_angle)
    if angle < 1e-6:
        return np.eye(3)  # 如果角度接近零，返回单位矩阵
    axis = axis_angle / angle
    # 使用Rodrigues公式
    return cv2.Rodrigues(axis_angle)[0]

# 计算旋转矩阵
R1 = axis_angle_to_rotation_matrix(rotation1)
R2 = axis_angle_to_rotation_matrix(rotation2)

# 计算相对旋转 R_1to2 = R2 * R1^T
R_1to2 = R2 @ R1.T

# 计算相对平移 t_1to2 = t2 - R_1to2 * t1
t_1to2 = translation2 - R_1to2 @ translation1

# 将相对旋转转回轴角表示
r_1to2 = cv2.Rodrigues(R_1to2)[0].flatten()

print("\n第一帧到第二帧的相对旋转:", r_1to2)
print("第一帧到第二帧的相对平移:", t_1to2)

# 显示图像（和原脚本一样）
def tensor_to_image(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    将归一化的图像张量转换为可显示的图像
    """
    # 确保张量在CPU上
    tensor = tensor.cpu().clone()
    
    # 如果是批次，只取第一个图像
    if tensor.dim() == 4:
        tensor = tensor[0]
    
    # 反归一化
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    
    # 将张量转换为numpy数组并调整通道顺序
    image = tensor.numpy()
    image = np.transpose(image, (1, 2, 0))
    
    # 确保值在[0, 1]范围内
    image = np.clip(image, 0, 1)
    
    return image

# 显示图像
img = tensor_to_image(frame1_tensor)
plt.figure(figsize=(10, 6))
plt.imshow(img)
plt.axis('off')
plt.title('Frame 1')
plt.show()

print("\n使用pose_set模型的测试完成！")