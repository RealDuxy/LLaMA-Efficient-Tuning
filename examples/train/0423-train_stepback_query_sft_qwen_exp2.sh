#!/bin/bash

# this is one task learning for with rephrase_query_generation dataset

# device: GPU: 4090 * 1

CUDA_VISIBLE_DEVICES=0 python  ../src/train_bash.py \
    --stage sft \
    --do_train \
    --training_mode emo \
    --dataset stepback_query_generation  \
    --template qwen \
    --dataset_dir ../data \
    --finetuning_type lora \
    --use_rslora False \
    --output_dir ../checkpoints/0423_stepback_query_qwen_exp1 \
    --preprocessing_num_workers 32 \
    --val_size 0.1 \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_target q_proj,v_proj,k_proj,o_proj,down_proj,up_proj,gate_proj \
    --logging_steps 10 \
    --save_steps 30 \
    --eval_steps 30 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --overwrite_output_dir \
    --cutoff_len 512  \
    --fp16  \
    --model_name_or_path /mnt/d/PycharmProjects/models/Qwen1.5-14B-Chat-GPTQ-Int4 \
