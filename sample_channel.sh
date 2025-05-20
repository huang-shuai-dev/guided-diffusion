#!/bin/bash

export OPENAI_LOGDIR=./sample_results  # 设置采样结果保存路径

# 设置模型参数
MODEL_FLAGS="--num_channels 128 --num_res_blocks 3 --image_height 16 --image_width 64 --in_channels 1"

# 设置扩散模型参数
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule linear"

# 设置采样参数
SAMPLE_FLAGS="--batch_size 32 --num_samples 160"

# 运行采样脚本
python scripts/channel_sample.py \
    --model_path ./train_results/model200000.pt \
    $MODEL_FLAGS \
    $DIFFUSION_FLAGS \
    $SAMPLE_FLAGS 