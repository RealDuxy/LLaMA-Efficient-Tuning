#!/bin/bash

#accelerate config # 首先配置分布式环境

CUDA_VISIBLE_DEVICES=0 python  ../src/train_bash.py \
    --stage sft \
    --do_train \
    --template chatglm3 \
    --dataset high_quality_qa,normal_qa,artical_interpre_qa \
    --dataset_dir ../data \
    --finetuning_type lora \
    --output_dir ../checkpoints/1119_stage1_spec_ft \
    --preprocessing_num_workers 64 \
    --overwrite_cache \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --lr_scheduler_type cosine \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_target all \
    --logging_steps 10 \
    --save_steps 500 \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --cutoff_len 2500  \
    --reserved_label_len 800  \
    --fp16  \
    --neft_alpha 5 \
    --model_name_or_path /root/chatglm3-6b \

CUDA_VISIBLE_DEVICES=0 python  ../src/train_bash.py \
    --stage sft \
    --do_train \
    --template chatglm3 \
    --dataset high_quality_qa,normal_qa,artical_interpre_qa,alpaca_gpt4_zh \
    --dataset_dir ../data \
    --finetuning_type lora \
    --output_dir ../checkpoints/1119_stage2_mix_alpaca_ft \
    --preprocessing_num_workers 64 \
    --overwrite_cache \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --lr_scheduler_type cosine \
    --checkpoint_dir ../checkpoints/1119_stage1_spec_ft \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_target all \
    --logging_steps 10 \
    --save_steps 500 \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --cutoff_len 2500  \
    --reserved_label_len 800  \
    --fp16  \
    --neft_alpha 5 \
    --model_name_or_path /root/chatglm3-6b \


CUDA_VISIBLE_DEVICES=0 python  ../src/train_bash.py \
    --stage sft \
    --do_train \
    --template chatglm3 \
    --dataset high_quality_qa,normal_qa,artical_interpre_qa,sharegpt_zh \
    --dataset_dir ../data \
    --finetuning_type lora \
    --output_dir ../checkpoints/1119_stage2_mix_sharegpt_ft \
    --preprocessing_num_workers 64 \
    --overwrite_cache \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --lr_scheduler_type cosine \
    --checkpoint_dir ../checkpoints/1119_stage1_spec_ft \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_target all \
    --logging_steps 10 \
    --save_steps 500 \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --cutoff_len 2500  \
    --reserved_label_len 800  \
    --fp16  \
    --neft_alpha 5 \
    --model_name_or_path /root/chatglm3-6b
