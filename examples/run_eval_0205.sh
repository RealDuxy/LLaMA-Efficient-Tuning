#!/bin/bash

# evaluate
CUDA_VISIBLE_DEVICES=0 python  ../src/train_bash.py \
    --stage sft \
    --do_eval \
    --template chatglm3-anan \
    --dataset askbob_qa \
    --max_samples 80 \
    --dataset_dir ../data \
    --output_dir  ../checkpoints/original \
    --split validation \
    --plot_loss \
    --cutoff_len 1700  \
    --per_device_train_batch_size 8 \
    --bf16  \
    --predict_with_generate \
    --model_name_or_path ../chatGLM3-6b \

# evaluate
CUDA_VISIBLE_DEVICES=0 python  ../src/train_bash.py \
    --stage sft \
    --do_eval \
    --template chatglm3-anan \
    --dataset askbob_qa \
    --max_samples 80 \
    --dataset_dir ../data \
    --finetuning_type lora \
    --adapter_name_or_path  ../checkpoints/0205_stage1_spec_ft/checkpoint-94 \
    --output_dir  ../checkpoints/0205_stage1_spec_ft/checkpoint-94 \
    --split validation \
    --plot_loss \
    --cutoff_len 1700  \
    --per_device_train_batch_size 8 \
    --bf16  \
    --predict_with_generate \
    --model_name_or_path ../chatGLM3-6b \

# evaluate
CUDA_VISIBLE_DEVICES=0 python  ../src/train_bash.py \
    --stage sft \
    --do_eval \
    --template chatglm3-anan \
    --dataset askbob_qa \
    --max_samples 80 \
    --dataset_dir ../data \
    --finetuning_type lora \
    --adapter_name_or_path  ../checkpoints/0205_stage1_spec_ft/checkpoint-188 \
    --output_dir  ../checkpoints/0205_stage1_spec_ft/checkpoint-188 \
    --split validation \
    --plot_loss \
    --cutoff_len 1700  \
    --per_device_train_batch_size 8 \
    --bf16  \
    --predict_with_generate \
    --model_name_or_path ../chatGLM3-6b \
