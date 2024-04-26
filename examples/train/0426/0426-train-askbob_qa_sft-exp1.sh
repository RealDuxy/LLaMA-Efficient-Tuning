#!/bin/bash

# This is multi-task learning with mixed training data in one-step training.



# exp1 qwen14b
# single gpus single  experiment
# askbob max_source_length does not have to be 3000, it's just for test.
CUDA_VISIBLE_DEVICES=0 python  ../../src/train_bash.py \
    --stage sft \
    --do_train \
    --template qwen \
    --dataset askbob_qa  \
    --dataset_dir ../data \
    --finetuning_type lora \
    --use_rslora False \
    --val_size 0.05 \
    --output_dir ../../checkpoints/qwen/0426_askbob_stage1_qwen14b_gptq_int4_exp1 \
    --preprocessing_num_workers 64 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --lr_scheduler_type cosine \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_target k_proj,o_proj,q_proj,v_proj,down_proj,gate_proj,up_proj \
    --logging_steps 10 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --overwrite_output_dir \
    --cutoff_len 3000  \
    --fp16 \
    --save_steps 562 \
    --eval_steps 562 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --model_name_or_path /mnt/e/UbuntuFiles/models_saved/Qwen1.5-14B-Chat-GPTQ-Int4
     >> 0426_sft.log 2>&1