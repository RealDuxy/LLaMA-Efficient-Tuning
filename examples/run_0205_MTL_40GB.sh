#!/bin/bash

#accelerate config # 首先配置分布式环境

CUDA_VISIBLE_DEVICES=0 python  ../src/train_bash.py \
    --stage sft \
    --do_train \
    --template chatglm3 \
    --dataset askbob_qa,sharegpt4_zh_only \
    --dataset_dir ../data \
    --finetuning_type lora \
    --output_dir ../checkpoints/0205_mtl_ft \
    --preprocessing_num_workers 32 \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --lr_scheduler_type cosine \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_target all \
    --logging_steps 10 \
    --save_steps 2000 \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --overwrite_output_dir \
    --cutoff_len 1700  \
    --reserved_label_len 700  \
    --bf16  \
    --model_name_or_path /root/autodl-tmp/chatglm3-6b \
    --neftune_noise_alpha 5 \

CUDA_VISIBLE_DEVICES=0 python  ../src/train_bash.py \
    --stage sft \
    --do_train \
    --template chatglm3 \
    --dataset askbob_qa \
    --dataset_dir ../data \
    --finetuning_type lora \
    --output_dir ../checkpoints/0205_ft \
    --preprocessing_num_workers 32 \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --lr_scheduler_type cosine \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_target all \
    --logging_steps 10 \
    --save_steps 100 \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --overwrite_output_dir \
    --cutoff_len 1700  \
    --reserved_label_len 700  \
    --bf16  \
    --model_name_or_path /root/autodl-tmp/chatglm3-6b \
    --neftune_noise_alpha 5 \
