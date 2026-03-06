#!/bin/bash
# 一条命令评估 CoTTA（clean 数据，严重级别 1）
python cotta_sqldepth.py --data_path ../autodl-tmp/data/kitti_c/clean/1/kitti_data --load_weights_folder ../autodl-tmp/KITTI_192x640_models --eval_split eigen --backbone resnet_lite --height 192 --width 640 --batch_size 1 --num_layers 50 --num_features 256 --model_dim 32 --patch_size 16 --dim_out 64 --query_nums 64 --min_depth 0.1 --max_depth 80.0 --cotta_steps 1 --mt_alpha 0.999 --rst_m 0.001 --ap 0.9 --num_aug 32 --lr 1e-5 --num_workers 0






