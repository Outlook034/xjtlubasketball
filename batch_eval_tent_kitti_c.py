import os
import subprocess
import numpy as np
import re
import sys
from datetime import datetime


log_filename = f"tent_tta_evaluation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
log_file = open(log_filename, 'w', encoding='utf-8')


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


original_stdout = sys.stdout
sys.stdout = Tee(sys.stdout, log_file)


corruptions = [
    "dark", "snow", "color_quant", "fog", "frost", "brightness", "contrast", "defocus_blur",
    "glass_blur", "motion_blur", "zoom_blur", "elastic_transform", "pixelate",
    "jpeg_compression", "shot_noise", "iso_noise", "impulse_noise", "gaussian_noise"
]
severities = [1, 2, 3, 4, 5]


base_args_list = [
    "--mode", "adapt",
    "--tta_mode", "tent",
    "--log_dir", "./logs",
    "--model_name", "sqldepth_tent",
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
    "--tent_steps", "1",
    "--tent_lr", "1e-4",
    "--tent_flip",
    "--tent_w_consistency", "1.0",
    "--tent_grad_clip", "1.0",
]


corruption_root = "../autodl-tmp/data/kitti_c"
results = {}

print(f"开始评估 Tent-style (regression) for SQLDepth，日志将保存到: {log_filename}")
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("使用脚本: Vectta.py")
print(f"TTA模式: {base_args_list[base_args_list.index('--tta_mode') + 1]}")


def _parse_metrics(output_text: str):
    match = re.search(
        r"&\s*([\d\.]+)\s*&\s*([\d\.]+)\s*&\s*([\d\.]+)\s*&\s*([\d\.]+)\s*&\s*([\d\.]+)\s*&\s*([\d\.]+)\s*&\s*([\d\.]+)",
        output_text,
    )
    if not match:
        return None
    return [float(match.group(i)) for i in range(1, 8)]




for corruption in corruptions:
    results[corruption] = []
    for severity in severities:
        data_path = os.path.join(corruption_root, corruption, str(severity), "kitti_data")
        if not os.path.exists(data_path):
            print(f"跳过 {data_path} (路径不存在)")
            continue

        all_args = ["python", "Vectta.py", "--data_path", data_path] + base_args_list

        print(f"\n评估 {corruption} 严重级别 {severity} ...")
        print(f"数据路径: {data_path}")

        try:
            process = subprocess.Popen(
                all_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
            )

            full_output = ""
            while True:
                output_line = process.stdout.readline()
                if output_line == '' and process.poll() is not None:
                    break
                if output_line:
                    print(output_line.strip())
                    full_output += output_line

            return_code = process.wait()
            if return_code != 0:
                print(f"警告: 进程以非零返回码 {return_code} 退出")

            metrics = _parse_metrics(full_output)
            if metrics is not None:
                results[corruption].append(metrics)
                print(f"成功解析指标: {metrics}")
            else:
                results[corruption].append([np.nan] * 7)
                print("未能解析指标，使用NaN值")

        except Exception as e:
            print(f"执行过程中出现错误: {e}")
            results[corruption].append([np.nan] * 7)


print("\n=== 各扰动平均结果 ===")
all_metrics = []
for corruption in corruptions:
    if results[corruption] and not all(np.isnan(np.array(results[corruption])).all(axis=1)):
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
    print("\n指标说明: abs_rel | sq_rel | rmse | rmse_log | a1 | a2 | a3")
else:
    print("\n=== 无有效结果可计算平均 ===")

print(f"\n评估完成，结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"日志已保存到: {log_filename}")

sys.stdout = original_stdout
log_file.close()
