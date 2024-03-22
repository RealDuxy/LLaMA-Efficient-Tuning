#!/bin/bash

# This is multi-task learning with mixed training data in one-step training.


# exp1 qwen14b

CUDA_VISIBLE_DEVICES=0 python  ../src/train_bash.py \
    --stage sft \
    --do_train \
    --template qwen \
    --dataset askbob_qa  \
    --dataset_dir ../data \
    --finetuning_type lora \
    --output_dir ../checkpoints/qwen/0320_askbob_stage1_qwen14b_gptq_int4 \
    --preprocessing_num_workers 64 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_target k_proj,o_proj,q_proj,v_proj,down_proj,gate_proj,up_proj \
    --logging_steps 10 \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --overwrite_output_dir \
    --cutoff_len 1700  \
    --fp16 \
    --save_steps 562 \
    --model_name_or_path /mnt/e/UbuntuFiles/models_saved/Qwen1.5-14B-Chat-GPTQ-Int4
     >> 0320_sft.log 2>&1

deepspeed --num_gpus=4  ../src/train_bash.py \
    --stage dpo \
    --do_train \
    --template chatglm3 \
    --dataset askbob_qa_comparison  \
    --dataset_dir ../data \
    --finetuning_type lora \
    --output_dir ../checkpoints/0322_chatglm3-stage1_dpo \
    --adapter_name_or_path  ../checkpoints/0205_stage1_spec_ft \
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
    --num_train_epochs 1.0 \
    --plot_loss \
    --overwrite_output_dir \
    --cutoff_len 1700  \
    --fp16 \
    --model_name_or_path /mnt/e/UbuntuFiles/models_saved/chatglm3/ \
    --deepspeed ../examples/train/v100_ds_config.json
     >> 0320_dpo.log 2>&1