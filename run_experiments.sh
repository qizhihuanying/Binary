#!/usr/bin/env bash

# 创建基本目录
mkdir -p logs/multilingual-e5-base
mkdir -p project/models/binary_head

GPU_IDS=(1 1 1 1 1 2 2 2 2 2 3 3 3 3 3 4 4 4 4 4 7 7 7 7 7)
LRS=(1e-6 3e-6 6e-6 1e-7 3e-7 6e-7 1e-5)
L2S=(0 1e-7 3e-7 6e-7 1e-6 3e-6 6e-6)
EPOCHS=(10 30 50 100)

NUM_GPUS=${#GPU_IDS[@]}
index=0


for epoch in "${EPOCHS[@]}"; do
    for lr in "${LRS[@]}"; do
        for l2 in "${L2S[@]}"; do

            while [ "$(jobs -p | wc -l)" -ge "$NUM_GPUS" ]; do
                sleep 1
            done

            gpu_id=${GPU_IDS[$((index % NUM_GPUS))]}
            log_filename="epoch=${epoch}+lr=${lr}+l2=${l2}.log"

            python main.py \
                --local_model_names intfloat/multilingual-e5-base \
                --langs zh \
                --use_binary_head \
                --epochs "$epoch" \
                --lr "$lr" \
                --l2 "$l2" \
                --batch_size 32 \
                --train_sample_ratio 1.0 \
                --val_ratio 0.2 \
                --test_ratio 1.0 \
                --device "$gpu_id" \
                --log_dir logs/multilingual-e5-base \
                --log_file "$log_filename" \
                --output_dir "project/models/binary_head/multilingual-e5-base" \
                --model_name_with_params \
            &

            ((index++))
        done
    done
done

wait

echo "All experiments completed!"
