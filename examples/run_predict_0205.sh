#!/bin/bash
sleep 2h
## evaluate
#CUDA_VISIBLE_DEVICES=0 python  ../src/train_bash.py \
#    --stage sft \
#    --do_predict \
#    --template chatglm3-anan \
#    --dataset askbob_qa \
#    --max_samples 800 \
#    --dataset_dir ../data \
#    --output_dir  ../checkpoints/original \
#    --split validation \
#    --plot_loss \
#    --cutoff_len 1700  \
#    --per_device_eval_batch_size 8 \
#    --bf16  \
#    --predict_with_generate \
#    --model_name_or_path ../chatGLM3-6b \
#
## evaluate
#CUDA_VISIBLE_DEVICES=0 python  ../src/train_bash.py \
#    --stage sft \
#    --do_predict \
#    --template chatglm3-anan \
#    --dataset askbob_qa \
#    --max_samples 800 \
#    --dataset_dir ../data \
#    --finetuning_type lora \
#    --adapter_name_or_path  ../checkpoints/0205_stage1_spec_ft/checkpoint-94 \
#    --output_dir  ../checkpoints/0205_stage1_spec_ft/checkpoint-94 \
#    --split validation \
#    --plot_loss \
#    --cutoff_len 1700  \
#    --per_device_eval_batch_size 8 \
#    --bf16  \
#    --predict_with_generate \
#    --model_name_or_path ../chatGLM3-6b \
#
## evaluate
#CUDA_VISIBLE_DEVICES=0 python  ../src/train_bash.py \
#    --stage sft \
#    --do_predict \
#    --template chatglm3-anan \
#    --dataset askbob_qa \
#    --max_samples 800 \
#    --dataset_dir ../data \
#    --finetuning_type lora \
#    --adapter_name_or_path  ../checkpoints/0205_stage1_spec_ft/checkpoint-188 \
#    --output_dir  ../checkpoints/0205_stage1_spec_ft/checkpoint-188 \
#    --split validation \
#    --plot_loss \
#    --cutoff_len 1700  \
#    --per_device_eval_batch_size 8 \
#    --bf16  \
#    --predict_with_generate \
#    --model_name_or_path ../chatGLM3-6b \

## evaluate
#CUDA_VISIBLE_DEVICES=0 python  ../src/train_bash.py \
#    --stage sft \
#    --do_predict \
#    --template chatglm3-anan \
#    --dataset askbob_qa \
#    --max_samples 800 \
#    --dataset_dir ../data \
#    --finetuning_type lora \
#    --adapter_name_or_path  ../checkpoints/0205_stage1_spec_ft/checkpoint-282 \
#    --output_dir  ../checkpoints/0205_stage1_spec_ft/checkpoint-282 \
#    --split validation \
#    --plot_loss \
#    --cutoff_len 1700  \
#    --per_device_eval_batch_size 8 \
#    --bf16  \
#    --predict_with_generate \
#    --model_name_or_path ../chatGLM3-6b \

## evaluate
#CUDA_VISIBLE_DEVICES=0 python  ../src/train_bash.py \
#    --stage sft \
#    --do_predict \
#    --template chatglm3-anan \
#    --dataset askbob_qa \
#    --max_samples 800 \
#    --dataset_dir ../data \
#    --finetuning_type lora \
#    --adapter_name_or_path  ../checkpoints/0205_mtl_ft/checkpoint-696 \
#    --output_dir  ../checkpoints/0205_mtl_ft/checkpoint-696 \
#    --split validation \
#    --plot_loss \
#    --cutoff_len 1700  \
#    --per_device_eval_batch_size 8 \
#    --bf16  \
#    --predict_with_generate \
#    --model_name_or_path ../chatGLM3-6b \
#
## evaluate
#CUDA_VISIBLE_DEVICES=0 python  ../src/train_bash.py \
#    --stage sft \
#    --do_predict \
#    --template chatglm3-anan \
#    --dataset askbob_qa \
#    --max_samples 800 \
#    --dataset_dir ../data \
#    --finetuning_type lora \
#    --adapter_name_or_path  ../checkpoints/0205_mtl_ft/checkpoint-1392 \
#    --output_dir  ../checkpoints/0205_mtl_ft/checkpoint-1392 \
#    --split validation \
#    --plot_loss \
#    --cutoff_len 1700  \
#    --per_device_eval_batch_size 8 \
#    --bf16  \
#    --predict_with_generate \
#    --model_name_or_path ../chatGLM3-6b \
#
## evaluate
#CUDA_VISIBLE_DEVICES=0 python  ../src/train_bash.py \
#    --stage sft \
#    --do_predict \
#    --template chatglm3-anan \
#    --dataset askbob_qa \
#    --max_samples 800 \
#    --dataset_dir ../data \
#    --finetuning_type lora \
#    --adapter_name_or_path  ../checkpoints/0205_mtl_ft/checkpoint-2088 \
#    --output_dir  ../checkpoints/0205_mtl_ft/checkpoint-2088 \
#    --split validation \
#    --plot_loss \
#    --cutoff_len 1700  \
#    --per_device_eval_batch_size 8 \
#    --bf16  \
#    --predict_with_generate \
#    --model_name_or_path ../chatGLM3-6b


## evaluate
#CUDA_VISIBLE_DEVICES=0 python  ../src/train_bash.py \
#    --stage sft \
#    --do_predict \
#    --template chatglm3-anan \
#    --dataset askbob_qa \
#    --max_samples 800 \
#    --dataset_dir ../data \
#    --finetuning_type lora \
#    --adapter_name_or_path  ../checkpoints/0205_stage2_mix_sharegpt_ft/checkpoint-696 \
#    --output_dir  ../checkpoints/0205_stage2_mix_sharegpt_ft/checkpoint-696 \
#    --split validation \
#    --plot_loss \
#    --cutoff_len 1700  \
#    --per_device_eval_batch_size 8 \
#    --bf16  \
#    --predict_with_generate \
#    --model_name_or_path ../chatGLM3-6b


## evaluate
#CUDA_VISIBLE_DEVICES=0 python  ../src/train_bash.py \
#    --stage sft \
#    --do_predict \
#    --template chatglm3-anan \
#    --dataset askbob_qa \
#    --max_samples 800 \
#    --dataset_dir ../data \
#    --finetuning_type lora \
#    --adapter_name_or_path  ../checkpoints/0205_stage2_mix_sharegpt_ft/checkpoint-2088 \
#    --output_dir  ../checkpoints/0205_stage2_mix_sharegpt_ft/checkpoint-2088 \
#    --split validation \
#    --plot_loss \
#    --cutoff_len 1700  \
#    --per_device_eval_batch_size 8 \
#    --bf16  \
#    --predict_with_generate \
#    --model_name_or_path ../chatGLM3-6b
#
## evaluate
#CUDA_VISIBLE_DEVICES=0 python  ../src/train_bash.py \
#    --stage sft \
#    --do_predict \
#    --template chatglm3-anan \
#    --dataset askbob_qa \
#    --max_samples 800 \
#    --dataset_dir ../data \
#    --finetuning_type lora \
#    --adapter_name_or_path  ../checkpoints/0205_stage2_mix_sharegpt_ft/checkpoint-1392 \
#    --output_dir  ../checkpoints/0205_stage2_mix_sharegpt_ft/checkpoint-1392 \
#    --split validation \
#    --plot_loss \
#    --cutoff_len 1700  \
#    --per_device_eval_batch_size 8 \
#    --bf16  \
#    --predict_with_generate \
#    --model_name_or_path ../chatGLM3-6b

# evaluate
CUDA_VISIBLE_DEVICES=0 python  ../src/train_bash.py \
    --stage sft \
    --do_predict \
    --template chatglm3-anan \
    --dataset sharegpt_zh_38k \
    --dataset_dir ../data \
    --output_dir  ../checkpoints/chatglm3_sharegpt \
    --per_device_eval_batch_size 8 \
    --bf16  \
    --predict_with_generate \
    --model_name_or_path ../chatGLM3-6b
