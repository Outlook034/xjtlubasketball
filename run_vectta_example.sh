#!/bin/bash

# VECTTA集成运行示例

echo "=== VECTTA集成运行示例 ==="

# 检查必要文件是否存在
if [ ! -f "evaluate_cityscapes_vectta.py" ]; then
    echo "错误: evaluate_cityscapes_vectta.py 文件不存在"
    exit 1
fi

if [ ! -f "args_vectta_cityscapes_eval.txt" ]; then
    echo "错误: args_vectta_cityscapes_eval.txt 配置文件不存在"
    exit 1
fi

echo "1. 生成clean伪标签 (第一步)"
echo "命令: python evaluate_cityscapes_vectta.py @args_generate_clean.txt"
echo ""

echo "2. 运行VECTTA评估 (使用配置文件)"
echo "命令: python evaluate_cityscapes_vectta.py @args_vectta_cityscapes_eval.txt"
echo ""

echo "3. 运行VECTTA评估 (直接指定参数)"
echo "命令: python evaluate_cityscapes_vectta.py --eval_data_path data/CS_RAW/ --eval_split cityscapes --height 320 --width 1024 --model_dim 32 --patch_size 16 --dim_out 64 --query_nums 64 --min_depth 0.001 --max_depth 80.0 --load_weights_folder models/res50_cityscapes_320x1024/ --eval_mono --post_process --save_pred_disps --use_vectta --vectta_mode vec --clean_pred_path clean_pred_disps.npy --vec_steps 5 --vec_update_mode bn_decoder --vec_lr 1e-4 --vec_early_stop --vec_early_stop_patience 3 --vec_grad_clip 1.0"
echo ""

echo "4. 运行EATA模式"
echo "命令: python evaluate_cityscapes_vectta.py @args_vectta_cityscapes_eval.txt --vectta_mode eata"
echo ""

echo "5. 不使用VECTTA (原始模式)"
echo "命令: python evaluate_cityscapes_vectta.py @args_vectta_cityscapes_eval.txt --no-use_vectta"
echo ""

echo "注意: 请确保以下条件满足:"
echo "- 模型权重文件存在于指定路径"
echo "- 数据集路径正确"
echo "- 如果使用clean伪标签，确保clean_pred_disps.npy文件存在"
echo "- 确保有足够的GPU内存进行测试时适应"
