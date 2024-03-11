#!/bin/bash

# This is multi-task learning with mixed training data in one-step training.

#accelerate config #

# exp1
# DPO training
# vanilla chatglm3-6b
# epoch 3.0
#
deepspeed --num_gpus=4  ../src/train_bash.py \
    --stage dpo \
    --do_train \
    --template chatglm3-anan \
    --dataset askbob_0301_chatglm3-vanilla-glm4_comparision  \
    --dataset_dir ../data \
    --finetuning_type lora \
    --output_dir ../checkpoints/0308_vanilla_dpo \
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
    --deepspeed ../examples/0304/v100_ds_config.json \

## exp2 dpo from sft
#deepspeed --num_gpus=4  ../src/train_bash.py \
#    --stage dpo \
#    --do_train \
#    --template chatglm3 \
#    --dataset askbob_0301_chatglm3-stage1-ckpt282-glm4_comparision  \
#    --dataset_dir ../data \
#    --finetuning_type lora \
#    --output_dir ../checkpoints/0304_chatglm3-stage1-ckpt282_dpo \
#    --adapter_name_or_path  ../checkpoints/0205_stage1_spec_ft/checkpoint-282 \
#    --preprocessing_num_workers 64 \
#    --overwrite_cache \
#    --per_device_train_batch_size 1 \
#    --gradient_accumulation_steps 8 \
#    --lr_scheduler_type cosine \
#    --lora_rank 16 \
#    --lora_alpha 32 \
#    --lora_target all \
#    --logging_steps 10 \
#    --learning_rate 2e-5 \
#    --num_train_epochs 1.0 \
#    --plot_loss \
#    --overwrite_output_dir \
#    --cutoff_len 1700  \
#    --bf16  \
#    --model_name_or_path /root/autodl-tmp/chatglm3-6b \
#    --deepspeed ../examples/0304/ds_config.json \

# exp4 qwen-0.5B sft
#deepspeed --num_gpus=4  ../src/train_bash.py \
#    --stage sft \
#    --do_train \
#    --template qwen \
#    --dataset askbob_qa  \
#    --dataset_dir ../data \
#    --finetuning_type lora \
#    --output_dir ../checkpoints/0304_qwen0-5B_stage1_spec_ft \
#    --preprocessing_num_workers 64 \
#    --overwrite_cache \
#    --per_device_train_batch_size 4 \
#    --gradient_accumulation_steps 2 \
#    --lr_scheduler_type cosine \
#    --lora_rank 16 \
#    --lora_alpha 32 \
#    --lora_target all \
#    --logging_steps 10 \
#    --save_steps 186  \
#    --learning_rate 2e-5 \
#    --num_train_epochs 3.0 \
#    --plot_loss \
#    --overwrite_output_dir \
#    --cutoff_len 1700  \
#    --bf16  \
#    --model_name_or_path /root/autodl-tmp/Qwen1.5-0.5B-Chat \
#    --deepspeed ../examples/0304/ds_config.json
#
## exp5 qwen-14B sft
#deepspeed --num_gpus=4  ../src/train_bash.py \
#    --stage sft \
#    --do_train \
#    --template qwen \
#    --dataset askbob_qa  \
#    --dataset_dir ../data \
#    --finetuning_type lora \
#    --output_dir ../checkpoints/0304_qwen14B_stage1_spec_ft \
#    --preprocessing_num_workers 64 \
#    --overwrite_cache \
#    --per_device_train_batch_size 1 \
#    --gradient_accumulation_steps 8 \
#    --lr_scheduler_type cosine \
#    --lora_rank 16 \
#    --lora_alpha 32 \
#    --lora_target all \
#    --logging_steps 10 \
#    --save_steps 186  \
#    --learning_rate 2e-5 \
#    --num_train_epochs 3.0 \
#    --plot_loss \
#    --overwrite_output_dir \
#    --cutoff_len 1700  \
#    --bf16  \
#    --model_name_or_path /root/autodl-tmp/Qwen1.5-14B-Chat \
#    --deepspeed ../examples/0304/ds_config.json
#
## exp3 qwen sft
#deepspeed --num_gpus=4  ../src/train_bash.py \
#    --stage sft \
#    --do_train \
#    --template qwen \
#    --dataset askbob_qa  \
#    --dataset_dir ../data \
#    --finetuning_type lora \
#    --output_dir ../checkpoints/0304_qwen7B_stage1_spec_ft \
#    --preprocessing_num_workers 64 \
#    --overwrite_cache \
#    --per_device_train_batch_size 1 \
#    --gradient_accumulation_steps 8 \
#    --lr_scheduler_type cosine \
#    --lora_rank 16 \
#    --lora_alpha 32 \
#    --lora_target all \
#    --logging_steps 10 \
#    --save_steps 186  \
#    --learning_rate 2e-5 \
#    --num_train_epochs 3.0 \
#    --plot_loss \
#    --overwrite_output_dir \
#    --cutoff_len 1700  \
#    --bf16  \
#    --model_name_or_path /root/autodl-tmp/Qwen1.5-7B-Chat \
#    --deepspeed ../examples/0304/ds_config.json \





