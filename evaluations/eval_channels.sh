#!/bin/bash

# 设置环境变量
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 设置参数
REF_BATCH="../../improved-diffusion/datasets/diffusion_1234.mat"  # 训练数据集路径（.mat格式）
SAMPLE_BATCH="../sample_results/samples_160x64x2.npz"  # 生成的样本路径（.npz格式）
OUTPUT_DIR="eval_results"  # 评估结果输出目录

# 运行评估脚本
python channel_eval.py \
    --ref_batch $REF_BATCH \
    --sample_batch $SAMPLE_BATCH \
    --output_dir $OUTPUT_DIR 