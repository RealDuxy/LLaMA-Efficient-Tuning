#!/bin/bash

#accelerate config # 首先配置分布式环境

CUDA_VISIBLE_DEVICES=0 python  ../src/train_bash.py \
    --stage sft \
    --do_train \
    --template chatglm3 \
    --dataset high_quality_qa,normal_qa,artical_interpre_qa,askbob_qa \
    --sample_ratio 1,1,1,1,1 \
    --dataset_dir ../data \
    --finetuning_type lora \
    --output_dir ../checkpoints/0107_stage1_spec_ft \
    --preprocessing_num_workers 64 \
    --overwrite_cache \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_target all \
    --logging_steps 10 \
    --save_steps 500 \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --overwrite_output_dir \
    --cutoff_len 2500  \
    --reserved_label_len 800  \
    --fp16  \
    --neftune_noise_alpha 5 \
    --model_name_or_path THUDM/chatglm3-6b \


CUDA_VISIBLE_DEVICES=0 python  ../src/train_bash.py \
    --stage sft \
    --do_train \
    --template chatglm3 \
    --dataset high_quality_qa,normal_qa,artical_interpre_qa,askbob_qa,alpaca_gpt4_zh,self_cognition \
    --sample_ratio 0.0156,0.0156,0.0156,0.0156,1.0,1.0 \
    --dataset_dir ../data \
    --finetuning_type lora \
    --output_dir ../checkpoints/0107_stage2_mix_alpaca_ft \
    --preprocessing_num_workers 64 \
    --overwrite_cache \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --adapter_name_or_path ../checkpoints/0107_stage1_spec_ft \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_target all \
    --logging_steps 10 \
    --save_steps 500 \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --overwrite_output_dir \
    --cutoff_len 2500  \
    --reserved_label_len 800  \
    --fp16  \
    --neftune_noise_alpha 5 \
    --model_name_or_path THUDM/chatglm3-6b \
