#!/bin/bash
# evaluate

CUDA_VISIBLE_DEVICES=0 python  ../src/train_bash.py \
    --stage sft \
    --do_predict \
    --template qwen \
    --dataset askbob_qa \
    --overwrite_cache \
    --preprocessing_num_workers 64 \
    --max_samples 7000 \
    --dataset_dir ../data \
    --output_dir  ../prediction_outputs/qwen1.5-0.5B-chat_output \
    --split validation \
    --plot_loss \
    --cutoff_len 1700  \
    --per_device_eval_batch_size 16 \
    --bf16  \
    --predict_with_generate \
    --model_name_or_path /mnt/d/PycharmProjects/models/Qwen1.5-0.5B-Chat \


