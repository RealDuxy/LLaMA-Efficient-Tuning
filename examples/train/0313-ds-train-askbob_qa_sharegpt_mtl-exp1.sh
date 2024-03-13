#!/bin/bash

# This is multi-task learning with mixed training data in one-step training.


# exp2 dpo from sft
deepspeed --num_gpus=4  ../src/train_bash.py \
    --stage sft \
    --do_train \
    --template chatglm3 \
    --dataset askbob_qa,alpaca_gpt4_zh  \
    --max_samples 30000 \
    --dataset_dir ../data \
    --finetuning_type lora \
    --output_dir ../checkpoints/0313_askbob_alpaca_gpt4_zh_mtl \
    --preprocessing_num_workers 64 \
    --overwrite_cache \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_target all \
    --logging_steps 10 \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --overwrite_output_dir \
    --cutoff_len 1700  \
    --fp16 \
    --model_name_or_path /mnt/e/UbuntuFiles/models_saved/chatglm3/ \
    --deepspeed ../examples/train/v100_ds_config.json \


deepspeed --num_gpus=4  ../src/train_bash.py \
    --stage sft \
    --do_train \
    --template chatglm3 \
    --dataset askbob_qa,alpaca_chatglm3_zh  \
    --max_samples 30000 \
    --dataset_dir ../data \
    --finetuning_type lora \
    --output_dir ../checkpoints/0313_askbob_alpaca_chatglm3_zh_mtl \
    --preprocessing_num_workers 64 \
    --overwrite_cache \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_target all \
    --logging_steps 10 \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --overwrite_output_dir \
    --cutoff_len 1700  \
    --fp16 \
    --model_name_or_path /mnt/e/UbuntuFiles/models_saved/chatglm3/ \
    --deepspeed ../examples/train/v100_ds_config.json \


