#!/bin/bash

# This is multi-task learning with mixed training data in one-step training.

#accelerate config #

# exp1 DPO training from vanilla chatglm3-6b
CUDA_VISIBLE_DEVICES=0 python  ../src/train_bash.py \
    --stage dpo \
    --do_train \
    --template chatglm3 \
    --dataset askbob_0301_chatglm3-vanilla-glm4_comparision  \
    --dataset_dir ../data \
    --finetuning_type lora \
    --output_dir ../checkpoints/0304_vanilla_dpo \
    --preprocessing_num_workers 64 \
    --overwrite_cache \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_target all \
    --logging_steps 10 \
    --learning_rate 2e-5 \
    --num_train_epochs 1.0 \
    --plot_loss \
    --overwrite_output_dir \
    --cutoff_len 1700  \
    --bf16  \
    --model_name_or_path /root/autodl-tmp/chatglm3-6b \
    --max_samples 100 \

# exp2 dpo from sft
CUDA_VISIBLE_DEVICES=0 python  ../src/train_bash.py \
    --stage dpo \
    --do_train \
    --template chatglm3 \
    --dataset askbob_0301_chatglm3-stage1-ckpt282-glm4_comparision  \
    --dataset_dir ../data \
    --finetuning_type lora \
    --output_dir ../checkpoints/0304_chatglm3-stage1-ckpt282_dpo \
    --adapter_name_or_path  ../checkpoints/0205_stage1_spec_ft/checkpoint-282 \
    --preprocessing_num_workers 64 \
    --overwrite_cache \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_target all \
    --logging_steps 10 \
    --learning_rate 2e-5 \
    --num_train_epochs 1.0 \
    --plot_loss \
    --overwrite_output_dir \
    --cutoff_len 1700  \
    --bf16  \
    --model_name_or_path /root/autodl-tmp/chatglm3-6b \
    --max_samples 100 \
