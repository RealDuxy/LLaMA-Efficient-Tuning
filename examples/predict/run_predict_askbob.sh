#!/bin/bash
# evaluate
CUDA_VISIBLE_DEVICES=0 python  ../src/train_bash.py \
    --stage sft \
    --do_predict \
    --template chatglm3-anan \
    --dataset askbob_qa \
    --max_samples 7000 \
    --dataset_dir ../data \
    --output_dir  ../prediction_outputs/chatglm3_vanilla_output \
    --split validation \
    --plot_loss \
    --cutoff_len 1700  \
    --per_device_eval_batch_size 5 \
    --bf16  \
    --predict_with_generate \
    --model_name_or_path /root/autodl-tmp/chatglm3-6b \

# evaluate
CUDA_VISIBLE_DEVICES=0 python  ../src/train_bash.py \
    --stage sft \
    --do_predict \
    --template chatglm3-anan \
    --dataset askbob_qa \
    --max_samples 7000 \
    --dataset_dir ../data \
    --finetuning_type lora \
    --adapter_name_or_path  ../checkpoints/0205_stage1_spec_ft/checkpoint-94 \
    --output_dir  ../prediction_outputs/chatglm3_stage1_output \
    --split validation \
    --plot_loss \
    --cutoff_len 1700  \
    --per_device_eval_batch_size 4 \
    --bf16  \
    --predict_with_generate \
    --model_name_or_path /root/autodl-tmp/chatglm3-6b \
