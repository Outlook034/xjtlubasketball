# import os
# import subprocess
# import numpy as np
# import re

# corruptions = [
#     "dark", "snow", "color_quant", "fog", "frost", "brightness", "contrast", "defocus_blur",
#     "glass_blur", "motion_blur", "zoom_blur", "elastic_transform", "pixelate",
#     "jpeg_compression", "shot_noise", "iso_noise", "impulse_noise", "gaussian_noise"
# ]
# severities = [1, 2, 3, 4, 5]

# # base_args_list = [
# #     "--log_dir /mnt/bn/videoarc-depthestimation-disk1/wangyouhong/exps/tmp/lr_res_kitti",
# #     "--model_name res_088",
# #     "--dataset kitti",
# #     "--eval_split eigen",
# #     "--backbone resnet_lite",
# #     "--height 192",
# #     "--width 640",
# #     "--batch_size 16",
# #     "--num_epochs 25",
# #     "--scheduler_step_size 15",
# #     "--num_layers 50",
# #     "--num_features 256",
# #     "--model_dim 32",
# #     "--patch_size 16",
# #     "--dim_out 64",
# #     "--query_nums 64",
# #     "--min_depth 0.001",
# #     "--max_depth 80.0",
# #     "--pretrained_pose",
# #     "--pose_net_path ../autodl-tmp/KITTI_192x640_models/",
# #     "--diff_lr",
# #     "--use_stereo",
# #     "--load_weights_folder ../autodl-tmp/KITTI_192x640_models",
# #     "--eval_mono",
# #     "--post_process"
# # ]

# # '''加伪标签做自适应'''
# # base_args_list = [
# #     "--mode adapt",
# #     # "--data_path ../autodl-tmp/data/kitti_c/target/kitti_data",  # 目标域路径，按需替换
# #     "--log_dir /mnt/bn/videoarc-depthestimation-disk1/wangyouhong/exps/tmp/lr_res_kitti",
# #     "--model_name res_088",
# #     "--dataset kitti",
# #     "--eval_split eigen",
# #     "--backbone resnet_lite",
# #     "--height 192",
# #     "--width 640",
# #     "--batch_size 16",
# #     "--num_layers 50",
# #     "--num_features 256",
# #     "--model_dim 32",
# #     "--patch_size 16",
# #     "--dim_out 64",
# #     "--query_nums 64",
# #     "--min_depth 0.001",
# #     "--max_depth 80.0",
# #     "--load_weights_folder ../autodl-tmp/KITTI_192x640_models",
# #     "--post_process"
# # ]

# base_args_list = [
    
# "--mode adapt", "--tta_mode vec",  "--log_dir /mnt/bn/videoarc-depthestimation-disk1/wangyouhong/exps/tmp/lr_res_kitti", "--model_name res_088", "--dataset kitti", "--eval_split eigen", "--backbone resnet_lite", "--height 192", "--width 640", "--batch_size 16", "--num_layers 50", "--num_features 256", "--model_dim 32", "--patch_size 16", "--dim_out 64", "--query_nums 64", "--min_depth 0.001", "--max_depth 80.0", "--load_weights_folder ../autodl-tmp/KITTI_192x640_models", "--clean_pred_path clean_pred_disps.npy", "--vec_steps 10", "--vec_update_mode all", "--vec_lr 1e-4", "--vec_early_stop", "--vec_early_stop_patience 3", "--vec_grad_clip 1.0" ]


# corruption_root = "../autodl-tmp/data/kitti_c"
# results = {}

# for corruption in corruptions:
#     results[corruption] = []
#     for severity in severities:
#         data_path = os.path.join(corruption_root, corruption, str(severity), "kitti_data")
#         if not os.path.exists(data_path):
#             print(f"Skip {data_path} (not found)")
#             continue
#         # 拼接一行命令行参数
#         # all_args = ["python", "eata_eval_kitti.py", f"--data_path", data_path] + base_args_list
                
#         all_args = ["python", "Vectta.py", f"--data_path", data_path] + base_args_list
#         print(f"Evaluating {corruption} severity {severity} ...")
#         # 运行评测
#         output = subprocess.getoutput(" ".join(all_args))
#         print(output)
#         # 解析输出
#         match = re.search(r"&\s*([\d\.]+)\s*&\s*([\d\.]+)\s*&\s*([\d\.]+)\s*&\s*([\d\.]+)\s*&\s*([\d\.]+)\s*&\s*([\d\.]+)\s*&\s*([\d\.]+)", output)
#         if match:
#             metrics = [float(match.group(i)) for i in range(1, 8)]
#             results[corruption].append(metrics)
#         else:
#             results[corruption].append([np.nan]*7)

# # 汇总与平均
# print("\n=== 各扰动平均结果 ===")
# all_metrics = []
# for corruption in corruptions:
#     arr = np.array(results[corruption])
#     mean = np.nanmean(arr, axis=0)
#     all_metrics.append(mean)
#     print(f"{corruption:16s}: " + " | ".join([f"{m:6.3f}" for m in mean]))

# overall = np.nanmean(np.array(all_metrics), axis=0)
# print("\n=== 所有扰动平均 ===")
# print(" | ".join([f"{m:6.3f}" for m in overall]))




import os
import subprocess
import numpy as np
import re
import sys
from datetime import datetime

# 设置日志文件
log_filename = f"evaluation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
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

corruptions = [
    "dark", "snow", "color_quant", "fog", "frost", "brightness", "contrast", "defocus_blur",
    "glass_blur", "motion_blur", "zoom_blur", "elastic_transform", "pixelate",
    "jpeg_compression", "shot_noise", "iso_noise", "impulse_noise", "gaussian_noise"
]
severities = [1, 2, 3, 4, 5]

# 正确格式化的参数列表
base_args_list = [
    "--mode", "adapt", 
    "--tta_mode", "vec",  
    "--log_dir", "/mnt/bn/videoarc-depthestimation-disk1/wangyouhong/exps/tmp/lr_res_kitti",
    "--model_name", "res_088", 
    "--dataset", "kitti", 
    "--eval_split", "eigen", 
    "--backbone", "resnet_lite", 
    "--height", "192", 
    "--width", "640", 
    "--batch_size", "16", 
    "--num_layers", "50", 
    "--num_features", "256", 
    "--model_dim", "32", 
    "--patch_size", "16", 
    "--dim_out", "64", 
    "--query_nums", "64", 
    "--min_depth", "0.001", 
    "--max_depth", "80.0", 
    "--load_weights_folder", "../autodl-tmp/KITTI_192x640_models", 
    "--clean_pred_path", "clean_pred_disps.npy", 
    "--vec_steps", "15", 
    "--vec_update_mode", "bn_decoder", 
    "--vec_lr", "1e-4", 
    "--vec_early_stop", 
    "--vec_early_stop_patience", "3", 
    "--vec_grad_clip", "1.0"
]

corruption_root = "../autodl-tmp/data/kitti_c"
results = {}

print(f"开始评估，日志将保存到: {log_filename}")
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

for corruption in corruptions:
    results[corruption] = []
    for severity in severities:
        data_path = os.path.join(corruption_root, corruption, str(severity), "kitti_data")
        if not os.path.exists(data_path):
            print(f"跳过 {data_path} (路径不存在)")
            continue
            
        # 构建命令参数列表 - 每个参数都是独立的列表元素
        all_args = ["python", "Vectta.py", "--data_path", data_path] + base_args_list
        
        print(f"\n评估 {corruption} 严重级别 {severity} ...")
        
        
        # 运行评测并实时捕获输出
        try:
            process = subprocess.Popen(
                all_args, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,  # 将标准错误重定向到标准输出
                universal_newlines=True,
                bufsize=1  # 行缓冲
            )
            
            # 实时输出并记录
            full_output = ""
            while True:
                output_line = process.stdout.readline()
                if output_line == '' and process.poll() is not None:
                    break
                if output_line:
                    print(output_line.strip())  # 实时输出到屏幕和日志文件
                    full_output += output_line
            
            # 获取返回码
            return_code = process.wait()
            if return_code != 0:
                print(f"警告: 进程以非零返回码 {return_code} 退出")
            
            # 解析输出
            match = re.search(r"&\s*([\d\.]+)\s*&\s*([\d\.]+)\s*&\s*([\d\.]+)\s*&\s*([\d\.]+)\s*&\s*([\d\.]+)\s*&\s*([\d\.]+)\s*&\s*([\d\.]+)", full_output)
            if match:
                metrics = [float(match.group(i)) for i in range(1, 8)]
                results[corruption].append(metrics)
                print(f"成功解析指标: {metrics}")
            else:
                results[corruption].append([np.nan]*7)
                print("未能解析指标，使用NaN值")
                
        except Exception as e:
            print(f"执行过程中出现错误: {e}")
            results[corruption].append([np.nan]*7)

# 汇总与平均
print("\n=== 各扰动平均结果 ===")
all_metrics = []
for corruption in corruptions:
    if results[corruption] and not all(np.isnan(metric).all() for metric in results[corruption]):  # 只处理有有效结果的扰动
        arr = np.array(results[corruption])
        mean = np.nanmean(arr, axis=0)
        all_metrics.append(mean)
        print(f"{corruption:16s}: " + " | ".join([f"{m:6.3f}" for m in mean]))
    else:
        print(f"{corruption:16s}: 无有效结果")

if all_metrics:
    overall = np.nanmean(np.array(all_metrics), axis=0)
    print("\n=== 所有扰动平均 ===")
    print(" | ".join([f"{m:6.3f}" for m in overall]))
else:
    print("\n=== 无有效结果可计算平均 ===")

print(f"\n评估完成，结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"日志已保存到: {log_filename}")

# 恢复标准输出并关闭日志文件
sys.stdout = original_stdout
log_file.close()