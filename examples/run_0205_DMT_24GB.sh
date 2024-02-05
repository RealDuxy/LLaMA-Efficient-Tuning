#!/bin/bash

# This is multi-task learning with mixed training data in one-step training.

#accelerate config #

CUDA_VISIBLE_DEVICES=0 python  ../src/train_bash.py \
    --stage sft \
    --do_train \
    --template chatglm3-anan \
    --dataset askbob_qa  \
    --dataset_dir ../data \
    --finetuning_type lora \
    --output_dir ../checkpoints/0205_stage1_spec_ft \
    --preprocessing_num_workers 16 \
    --overwrite_cache \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --lr_scheduler_type cosine \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_target all \
    --logging_steps 10 \
    --save_steps 372 \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --overwrite_output_dir \
    --cutoff_len 1700  \
    --reserved_label_len 700  \
    --bf16  \
    --model_name_or_path ../chatGLM3-6b \
    --neftune_noise_alpha 5 \

## evaluate
#CUDA_VISIBLE_DEVICES=0 python  ../src/train_bash.py \
#    --stage sft \
#    --do_eval \
#    --template chatglm3 \
#    --dataset askbob_qa \
#    --sample_ratio 0.05 \
#    --dataset_dir ../data \
#    --finetuning_type lora \
#    --adapter_name_or_path  ../checkpoints/0201_stage1_spec_ft_test \
#    --output_dir  ../checkpoints/0201_stage1_spec_ft_eval_test \
#    --split validation \
#    --plot_loss \
#    --cutoff_len 1700  \
#    --reserved_label_len 700  \
#    --per_device_train_batch_size 4 \
#    --bf16  \
#    --predict_with_generate \
#    --model_name_or_path ../chatGLM3-6b \
