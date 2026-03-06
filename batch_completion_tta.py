import argparse
import os
import subprocess
import re
from datetime import datetime
import numpy as np


def tee_print(log_file, message):
    print(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def parse_metrics(output):
    pattern = r"&\s*([\d\.NaNnan\-]+)\s*&\s*([\d\.NaNnan\-]+)\s*&\s*([\d\.NaNnan\-]+)\s*&\s*([\d\.NaNnan\-]+)\s*&\s*([\d\.NaNnan\-]+)\s*&\s*([\d\.NaNnan\-]+)\s*&\s*([\d\.NaNnan\-]+)"
    match = re.search(pattern, output)
    if match:
        metrics = []
        for i in range(1, 8):
            try:
                metrics.append(float(match.group(i)))
            except ValueError:
                metrics.append(np.nan)
        return metrics
    return None


def main():
    parser = argparse.ArgumentParser(description="Batch runner for completion_tta.py")
    parser.add_argument("--completion_script", type=str, default="completion_tta.py",
                        help="Path to completion_tta.py")
    parser.add_argument("--corruption_root", type=str, required=True,
                        help="Root directory containing corruption folders")
    parser.add_argument("--load_weights_folder", type=str, required=True,
                        help="SQLDepth weights folder (encoder.pth/depth.pth)")
    parser.add_argument("--log_dir", type=str, default=".",
                        help="Directory to store evaluation logs")
    parser.add_argument("--eval_split", type=str, default="eigen")
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--tta_steps", type=int, default=2)
    parser.add_argument("--tta_lr", type=float, default=4e-5)
    parser.add_argument("--tta_update_mode", type=str, choices=["bn", "bn_decoder", "all"], default="bn")
    parser.add_argument("--sparse_keep_ratio", type=float, default=0.1)
    parser.add_argument("--extra_args", nargs=argparse.REMAINDER, default=[],
                        help="Additional args passed to completion_tta.py")
    args = parser.parse_args()

    log_filename = os.path.join(
        args.log_dir,
        f"completion_tta_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )
    os.makedirs(args.log_dir, exist_ok=True)
    log_file = open(log_filename, "w", encoding="utf-8")

    tee_print(log_file, f"开始 Completion-TTA 评估，日志: {log_filename}")
    tee_print(log_file, f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    corruptions = [
        "dark", "snow", "color_quant", "fog", "frost", "brightness", "contrast",
        "defocus_blur", "glass_blur", "motion_blur", "zoom_blur", "elastic_transform",
        "pixelate", "jpeg_compression", "shot_noise", "iso_noise", "impulse_noise",
        "gaussian_noise"
    ]
    severities = [1, 2, 3, 4, 5]

    base_args = [
        "--load_weights_folder", args.load_weights_folder,
        "--eval_split", args.eval_split,
        "--height", str(args.height),
        "--width", str(args.width),
        "--batch_size", str(args.batch_size),
        "--tta_steps", str(args.tta_steps),
        "--tta_lr", str(args.tta_lr),
        "--tta_update_mode", args.tta_update_mode,
        "--sparse_keep_ratio", str(args.sparse_keep_ratio)
    ] + args.extra_args

    results = {}

    for corruption in corruptions:
        results[corruption] = []
        for severity in severities:
            data_path = os.path.join(
                args.corruption_root, corruption, str(severity), "kitti_data"
            )
            if not os.path.exists(data_path):
                tee_print(log_file, f"跳过 {data_path} (路径不存在)")
                continue

            tee_print(log_file, f"\n评估 {corruption} 严重级别 {severity} ...")
            cmd = ["python", args.completion_script, "--data_path", data_path] + base_args
            tee_print(log_file, "命令: " + " ".join(cmd))

            try:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1
                )

                full_output = ""
                for line in process.stdout:
                    line_stripped = line.rstrip()
                    tee_print(log_file, line_stripped)
                    full_output += line

                return_code = process.wait()
                if return_code != 0:
                    tee_print(log_file, f"警告: 进程以非零返回码 {return_code} 退出")

                metrics = parse_metrics(full_output)
                if metrics is None:
                    tee_print(log_file, "未能解析指标，使用 NaN")
                    metrics = [np.nan] * 7
                else:
                    tee_print(log_file, f"成功解析指标: {metrics}")
                results[corruption].append(metrics)

            except Exception as exc:
                tee_print(log_file, f"执行过程中出现错误: {exc}")
                results[corruption].append([np.nan] * 7)

    tee_print(log_file, "\n=== 各扰动平均结果 ===")
    all_metrics = []
    for corruption in corruptions:
        arr = np.array(results.get(corruption, []), dtype=float)
        if arr.size == 0:
            tee_print(log_file, f"{corruption:16s}: 无有效结果")
            continue
        mean_vals = np.nanmean(arr, axis=0)
        if np.all(np.isnan(mean_vals)):
            tee_print(log_file, f"{corruption:16s}: 无有效结果")
            continue
        all_metrics.append(mean_vals)
        tee_print(log_file, f"{corruption:16s}: " + " | ".join(f"{m:6.3f}" for m in mean_vals))

    if all_metrics:
        overall = np.nanmean(np.array(all_metrics), axis=0)
        tee_print(log_file, "\n=== 所有扰动平均 ===")
        tee_print(log_file, " | ".join(f"{m:6.3f}" for m in overall))
    else:
        tee_print(log_file, "\n=== 无有效结果可计算平均 ===")

    tee_print(log_file, f"\n评估完成，结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_file.close()


if __name__ == "__main__":
    main()
