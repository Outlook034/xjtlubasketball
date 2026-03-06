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
    parser = argparse.ArgumentParser(description="批量评估 SQLdepth SSA TTA 在所有 KITTI-C 扰动上")
    parser.add_argument("--corruption_root", type=str, required=True,
                        help="KITTI-C 数据集根目录")
    parser.add_argument("--load_weights_folder", type=str, required=True,
                        help="SQLDepth 权重文件夹 (encoder.pth/depth.pth)")
    parser.add_argument("--ssa_stats_path", type=str, required=True,
                        help="SSA 特征统计文件路径")
    parser.add_argument("--log_dir", type=str, default=".",
                        help="日志保存目录")
    parser.add_argument("--eval_split", type=str, default="eigen")
    parser.add_argument("--backbone", type=str, default="resnet_lite")
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=50)
    parser.add_argument("--num_features", type=int, default=256)
    parser.add_argument("--model_dim", type=int, default=32)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--dim_out", type=int, default=64)
    parser.add_argument("--query_nums", type=int, default=64)
    parser.add_argument("--min_depth", type=float, default=0.001)
    parser.add_argument("--max_depth", type=float, default=80.0)
    
    # SSA 参数
    parser.add_argument("--ssa_topk", type=int, default=64)
    parser.add_argument("--ssa_steps", type=int, default=1)
    parser.add_argument("--ssa_lr", type=float, default=1e-4)
    parser.add_argument("--ssa_update_mode", type=str,
                        choices=["encoder_bn", "encoder_all", "all"], default="encoder_bn")
    parser.add_argument("--ssa_feature_pool", type=str, choices=["avg", "max"], default="avg")
    parser.add_argument("--ssa_eps", type=float, default=1e-6)
    parser.add_argument("--ssa_weight_bias", type=float, default=1e-3)
    parser.add_argument("--ssa_weight_exp", type=float, default=1.0)
    parser.add_argument("--ssa_grad_clip", type=float, default=0.0)
    
    parser.add_argument("--extra_args", nargs=argparse.REMAINDER, default=[],
                        help="传递给 tta_sqldepth_kitti_c.py 的额外参数")
    args = parser.parse_args()

    # 检查 SSA 统计文件是否存在
    if not os.path.exists(args.ssa_stats_path):
        raise FileNotFoundError(f"SSA 统计文件不存在: {args.ssa_stats_path}\n"
                               f"请先运行 ssa_feature_stats.py 生成统计文件")

    log_filename = os.path.join(
        args.log_dir,
        f"ssa_tta_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )
    os.makedirs(args.log_dir, exist_ok=True)
    log_file = open(log_filename, "w", encoding="utf-8")

    tee_print(log_file, f"开始 SSA TTA 批量评估，日志: {log_filename}")
    tee_print(log_file, f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    tee_print(log_file, f"SSA 统计文件: {args.ssa_stats_path}")
    tee_print(log_file, f"模型权重: {args.load_weights_folder}")
    tee_print(log_file, f"KITTI-C 根目录: {args.corruption_root}")

    corruptions = [
        "dark", "snow", "color_quant", "fog", "frost", "brightness", "contrast",
        "defocus_blur", "glass_blur", "motion_blur", "zoom_blur", "elastic_transform",
        "pixelate", "jpeg_compression", "shot_noise", "iso_noise", "impulse_noise",
        "gaussian_noise"
    ]
    severities = [1, 2, 3, 4, 5]

    base_args = [
        "--mode", "adapt",
        "--tta_mode", "ssa",
        "--load_weights_folder", args.load_weights_folder,
        "--log_dir", args.log_dir,
        "--model_name", "sqldepth_ssa",
        "--dataset", "kitti",
        "--eval_split", args.eval_split,
        "--backbone", args.backbone,
        "--height", str(args.height),
        "--width", str(args.width),
        "--batch_size", str(args.batch_size),
        "--num_layers", str(args.num_layers),
        "--num_features", str(args.num_features),
        "--model_dim", str(args.model_dim),
        "--patch_size", str(args.patch_size),
        "--dim_out", str(args.dim_out),
        "--query_nums", str(args.query_nums),
        "--min_depth", str(args.min_depth),
        "--max_depth", str(args.max_depth),
        "--ssa_stats_path", args.ssa_stats_path,
        "--ssa_topk", str(args.ssa_topk),
        "--ssa_steps", str(args.ssa_steps),
        "--ssa_lr", str(args.ssa_lr),
        "--ssa_update_mode", args.ssa_update_mode,
        "--ssa_feature_pool", args.ssa_feature_pool,
        "--ssa_eps", str(args.ssa_eps),
        "--ssa_weight_bias", str(args.ssa_weight_bias),
        "--ssa_weight_exp", str(args.ssa_weight_exp),
        "--ssa_grad_clip", str(args.ssa_grad_clip),
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
            cmd = ["python", "tta_sqldepth_kitti_c.py", "--data_path", data_path] + base_args
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
        tee_print(log_file, "\n指标说明: abs_rel | sq_rel | rmse | rmse_log | a1 | a2 | a3")
    else:
        tee_print(log_file, "\n=== 无有效结果可计算平均 ===")

    tee_print(log_file, f"\n评估完成，结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_file.close()
    print(f"\n日志已保存到: {log_filename}")


if __name__ == "__main__":
    main()

