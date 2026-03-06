#!/bin/bash

IMAGE_ROOT="leftImg8bit/test"
TEST_LIST="splits/cityscapes/test_files.txt"
SAVE_PATH="data_cityscapes_corruptions"
H=320
W=1024
SEVERITY="1 2 3 4 5"  # 使用空格分隔，不是方括号

# 所有扰动类型
CORRUPTIONS=(
    "brightness" "dark" "fog" "frost" "snow" "contrast"
    "defocus_blur" "glass_blur" "motion_blur" "zoom_blur"
    "elastic" "color_quant" "gaussian_noise" "impulse_noise"
    "shot_noise" "iso_noise" "pixelate" "jpeg"
)

for corr in "${CORRUPTIONS[@]}"; do
    echo "Generating $corr corruptions..."
    python create_cityscapes_corruptions.py \
        --image_root "$IMAGE_ROOT" \
        --test_list "$TEST_LIST" \
        --folder_name leftImg8bit_sequence \
        --split test \
        --H $H \
        --W $W \
        --save_path "$SAVE_PATH" \
        --severity_levels $SEVERITY \
        --if_${corr}
    echo "Done with $corr"
    echo ""
done

# 复制干净图像
echo "Copying clean images..."
python create_cityscapes_corruptions.py \
    --image_root "$IMAGE_ROOT" \
    --test_list "$TEST_LIST" \
    --folder_name leftImg8bit_sequence \
    --split test \
    --H $H \
    --W $W \
    --save_path "$SAVE_PATH" \
    --if_copy_clean

echo "All corruptions generated!"

