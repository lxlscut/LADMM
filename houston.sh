#!/bin/bash

# 设置要替换的文件名
FILE_PATH="Houston.py"

# 定义随机种子列表
SEEDS=(7270 860 5390 5191 5734 6265 466 4426 5578 8322)

# 设置输出结果的目录
OUTPUT_DIR="result/houston0005"

# 创建输出目录（如果不存在的话）
mkdir -p $OUTPUT_DIR

# 循环遍历每个种子值
for SEED in "${SEEDS[@]}"; do
    # 使用sed命令查找并替换 setup_seed 后的任意数字为当前的 SEED 值
    sed -i "s/setup_seed([0-9]\+)/setup_seed($SEED)/" $FILE_PATH

    # 运行程序并将结果保存到相应的文件中
    python3 $FILE_PATH > "$OUTPUT_DIR/result_$SEED.txt"

    echo "Run completed for seed $SEED, result saved to $OUTPUT_DIR/result_$SEED.txt"
done
