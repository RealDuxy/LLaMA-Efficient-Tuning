#!/bin/bash

# This is multi-task learning with mixed training data in one-step training.

#accelerate config #

CUDA_VISIBLE_DEVICES=0 python  ../src/train_bash.py \
    --stage sft \
    --do_train \
    --dataset stepback_query_generation  \
    --template chatglm3 \
    --dataset_dir ../data \
    --finetuning_type lora \
    --output_dir ../checkpoints/stepback_query_generation_exp1 \
    --preprocessing_num_workers 32 \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_target all \
    --logging_steps 10 \
    --save_steps  80\
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --overwrite_output_dir \
    --cutoff_len 512  \
    --bf16  \
    --model_name_or_path /mnt/d/PycharmProjects/models/chatglm3-6b/ \
