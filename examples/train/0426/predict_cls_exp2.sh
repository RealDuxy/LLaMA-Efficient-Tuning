#!/bin/bash

MODEL_PATH=/mnt/e/UbuntuFiles/models_saved/Qwen1.5-14B-Chat-GPTQ-Int4
#MODEL_PATH=/mnt/d/PycharmProjects/models/Qwen1.5-14B-Chat-GPTQ-Int4

# exp1 qwen14b
# single gpus single  experiment
# askbob max_source_length does not have to be 3000, it's just for test.
CUDA_VISIBLE_DEVICES=1 python  ../src/train_bash.py \
    --stage sft \
    --do_predict \
    --template qwen \
    --dataset reference_classification_cot  \
    --dataset_dir ../data \
    --finetuning_type lora \
    --output_dir ../checkpoints/qwen/0428_reference_classification_cot_qwen14b_gptq_int4_exp2 \
    --adapter_name_or_path ../checkpoints/qwen/0428_reference_classification_cot_qwen14b_gptq_int4_exp2 \
    --logging_steps 5 \
    --overwrite_output_dir \
    --overwrite_cache \
    --cutoff_len 1300  \
    --preprocessing_num_workers 32 \
    --per_device_eval_batch_size 2 \
    --model_name_or_path $MODEL_PATH \
    --predict_with_generate
