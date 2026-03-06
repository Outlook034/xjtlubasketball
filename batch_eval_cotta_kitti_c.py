"""
CoTTA 综合评估脚本
在 KITTI-C 数据集上评估 CoTTA 在不同扰动和严重级别下的性能
"""
import os
import subprocess
import numpy as np
import re
import sys
from datetime import datetime

# 设置日志文件
log_filename = f"evaluation_log_cotta_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
log_file = open(log_filename, 'w')

# 重定向标准输出到文件和屏幕
class Tee:
    def __init__(self, *files):
        self.files = files
    
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # 确保实时写入
    
    def flush(self):
        for f in self.files:
            f.flush()

# 保存原始标准输出
original_stdout = sys.stdout
sys.stdout = Tee(sys.stdout, log_file)

# 所有扰动类型
corruptions = [
    "dark", "snow", "color_quant", "fog", "frost", "brightness", "contrast", "defocus_blur",
    "glass_blur", "motion_blur", "zoom_blur", "elastic_transform", "pixelate",
    "jpeg_compression", "shot_noise", "iso_noise", "impulse_noise", "gaussian_noise"
]

# 严重级别
severities = [1, 2, 3, 4, 5]

# CoTTA 基础参数列表
base_args_list = [
    "--eval_split", "eigen",
    "--backbone", "resnet_lite",
    "--height", "192",
    "--width", "640",
    "--batch_size", "1",  # CoTTA 建议使用 batch_size=1
    "--num_layers", "50",
    "--num_features", "256",
    "--model_dim", "32",
    "--patch_size", "16",
    "--dim_out", "64",
    "--query_nums", "64",
    "--min_depth", "0.1",
    "--max_depth", "80.0",
    "--load_weights_folder", "../autodl-tmp/KITTI_192x640_models",
    # CoTTA 参数
    "--cotta_steps", "1",
    "--mt_alpha", "0.999",
    "--rst_m", "0.001",
    "--ap", "0.9",
    "--num_aug", "32",
    "--lr", "1e-5",
    "--num_workers", "0",  # 便于调试
    # 可选：添加 --disable_cotta 来测试原始模型性能
]

# 扰动数据根目录
corruption_root = "../autodl-tmp/data/kitti_c"

results = {}

print(f"开始 CoTTA 评估，日志将保存到: {log_filename}")
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"评估脚本: cotta_sqldepth.py")
print(f"评估模式: CoTTA (Continual Test-Time Adaptation)")
print(f"="*80)

for corruption in corruptions:
    results[corruption] = []
    for severity in severities:
        data_path = os.path.join(corruption_root, corruption, str(severity), "kitti_data")
        
        if not os.path.exists(data_path):
            print(f"跳过 {corruption} severity {severity}: 路径不存在 - {data_path}")
            results[corruption].append([np.nan]*7)
            continue
        
        # 构建命令参数列表
        all_args = ["python", "cotta_sqldepth.py", "--data_path", data_path] + base_args_list
        
        print(f"\n{'='*80}")
        print(f"评估: {corruption} | 严重级别: {severity}")
        print(f"数据路径: {data_path}")
        print(f"{'='*80}")
        
        # 运行评测并实时捕获输出
        # 注意：不重定向 stderr，让 tqdm 可以输出到 stderr 并正确显示
        try:
            process = subprocess.Popen(
                all_args, 
                stdout=subprocess.PIPE,  # 只捕获 stdout
                stderr=None,  # stderr 直接输出到终端，不被重定向（这样 tqdm 可以正确显示）
                universal_newlines=True,
                bufsize=1  # 行缓冲
            )
            
            # 实时输出并记录（只处理 stdout，stderr 会直接显示在终端）
            full_output = ""
            while True:
                output_line = process.stdout.readline()
                if output_line == '' and process.poll() is not None:
                    break
                if output_line:
                    print(output_line.strip())  # 实时输出到屏幕和日志文件
                    full_output += output_line
            
            # 等待进程完成，确保所有输出完成
            return_code = process.wait()
            
            if return_code != 0:
                print(f"警告: 进程以非零返回码 {return_code} 退出")
                results[corruption].append([np.nan]*7)
                continue
            
            # 解析输出 - 查找评估指标
            # 格式: &   0.xxx  &   x.xxx  &   x.xxx  &   0.xxx  &   0.xxx  &   0.xxx  &   0.xxx  \\
            match = re.search(
                r"&\s*([\d\.]+)\s*&\s*([\d\.]+)\s*&\s*([\d\.]+)\s*&\s*([\d\.]+)\s*&\s*([\d\.]+)\s*&\s*([\d\.]+)\s*&\s*([\d\.]+)",
                full_output
            )
            
            if match:
                metrics = [float(match.group(i)) for i in range(1, 8)]
                results[corruption].append(metrics)
                print(f"✓ 成功解析指标: abs_rel={metrics[0]:.3f}, sq_rel={metrics[1]:.3f}, "
                      f"rmse={metrics[2]:.3f}, rmse_log={metrics[3]:.3f}, "
                      f"a1={metrics[4]:.3f}, a2={metrics[5]:.3f}, a3={metrics[6]:.3f}")
            else:
                # 尝试其他格式
                match2 = re.search(
                    r"abs_rel.*?(\d+\.\d+).*?sq_rel.*?(\d+\.\d+).*?rmse.*?(\d+\.\d+).*?rmse_log.*?(\d+\.\d+).*?a1.*?(\d+\.\d+).*?a2.*?(\d+\.\d+).*?a3.*?(\d+\.\d+)",
                    full_output,
                    re.DOTALL
                )
                if match2:
                    metrics = [float(match2.group(i)) for i in range(1, 8)]
                    results[corruption].append(metrics)
                    print(f"✓ 成功解析指标 (备用格式): {metrics}")
                else:
                    results[corruption].append([np.nan]*7)
                    print("✗ 未能解析指标，使用 NaN 值")
                    print(f"输出片段 (最后500字符):\n{full_output[-500:]}")
                
        except Exception as e:
            print(f"✗ 执行过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
            results[corruption].append([np.nan]*7)

# 汇总与平均
print(f"\n{'='*80}")
print("=== 各扰动平均结果 ===")
print(f"{'='*80}")
print(f"{'扰动类型':<20} | {'abs_rel':>8} | {'sq_rel':>8} | {'rmse':>8} | {'rmse_log':>8} | {'a1':>8} | {'a2':>8} | {'a3':>8}")
print("-" * 80)

all_metrics = []
for corruption in corruptions:
    if results[corruption] and not all(np.isnan(metric).all() for metric in results[corruption]):
        # 只处理有有效结果的扰动
        arr = np.array(results[corruption])
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        all_metrics.append(mean)
        print(f"{corruption:<20} | {mean[0]:8.3f} | {mean[1]:8.3f} | {mean[2]:8.3f} | "
              f"{mean[3]:8.3f} | {mean[4]:8.3f} | {mean[5]:8.3f} | {mean[6]:8.3f}")
    else:
        print(f"{corruption:<20} | {'N/A':>8} | {'N/A':>8} | {'N/A':>8} | "
              f"{'N/A':>8} | {'N/A':>8} | {'N/A':>8} | {'N/A':>8}")

if all_metrics:
    overall = np.nanmean(np.array(all_metrics), axis=0)
    overall_std = np.nanstd(np.array(all_metrics), axis=0)
    print("-" * 80)
    print(f"{'所有扰动平均':<20} | {overall[0]:8.3f} | {overall[1]:8.3f} | {overall[2]:8.3f} | "
          f"{overall[3]:8.3f} | {overall[4]:8.3f} | {overall[5]:8.3f} | {overall[6]:8.3f}")
    print(f"{'标准差':<20} | {overall_std[0]:8.3f} | {overall_std[1]:8.3f} | {overall_std[2]:8.3f} | "
          f"{overall_std[3]:8.3f} | {overall_std[4]:8.3f} | {overall_std[5]:8.3f} | {overall_std[6]:8.3f}")
else:
    print("\n=== 无有效结果可计算平均 ===")

# 按严重级别汇总
print(f"\n{'='*80}")
print("=== 按严重级别汇总 ===")
print(f"{'='*80}")
print(f"{'严重级别':<12} | {'abs_rel':>8} | {'sq_rel':>8} | {'rmse':>8} | {'rmse_log':>8} | {'a1':>8} | {'a2':>8} | {'a3':>8}")
print("-" * 80)

for severity in severities:
    severity_metrics = []
    for corruption in corruptions:
        if results[corruption] and len(results[corruption]) > severity - 1:
            metric = results[corruption][severity - 1]
            if not np.isnan(metric).all():
                severity_metrics.append(metric)
    
    if severity_metrics:
        arr = np.array(severity_metrics)
        mean = np.nanmean(arr, axis=0)
        print(f"Severity {severity:<8} | {mean[0]:8.3f} | {mean[1]:8.3f} | {mean[2]:8.3f} | "
              f"{mean[3]:8.3f} | {mean[4]:8.3f} | {mean[5]:8.3f} | {mean[6]:8.3f}")
    else:
        print(f"Severity {severity:<8} | {'N/A':>8} | {'N/A':>8} | {'N/A':>8} | "
              f"{'N/A':>8} | {'N/A':>8} | {'N/A':>8} | {'N/A':>8}")

print(f"\n{'='*80}")
print(f"评估完成，结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"日志已保存到: {log_filename}")
print(f"{'='*80}")

# 恢复标准输出并关闭日志文件
sys.stdout = original_stdout
log_file.close()

print(f"\n评估完成！结果已保存到: {log_filename}")

