#!/bin/bash

export OPENAI_LOGDIR=./train_results  # 设置训练结果保存路径

# 设置模型参数
MODEL_FLAGS="--num_channels 128 --num_res_blocks 3 --image_height 16 --image_width 64 "

# 设置扩散模型参数
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule linear"

# 设置训练参数
TRAIN_FLAGS="--lr 1e-4 --batch_size 32 --save_interval 10000 --log_interval 1000 --lr_anneal_steps 100000"

# 运行训练脚本
python scripts/channel_train.py \
    --data_dir ../improved-diffusion/datasets/diffusion_1234.mat \
    $MODEL_FLAGS \
    $DIFFUSION_FLAGS \
    $TRAIN_FLAGS 