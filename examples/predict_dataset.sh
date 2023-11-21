#!/bin/bash

echo "Hello there! pls wait 3 hours"
sleep 2h
echo "Oops! I fell asleep for a 3 hours!"


CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --model_name_or_path /mnt/d/PycharmProjects/models/chatglm3-6b \
    --do_predict \
    --dataset sharegpt_zh \
    --template chatglm3 \
    --output_dir data/sharegpt_zh/ \
    --per_device_eval_batch_size 8 \
    --predict_with_generate \

#accelerate launch src/train_bash.py \
#    --stage sft \
#    --model_name_or_path THUDM/chatglm3-6b \
#    --do_predict \
#    --dataset alpaca_gpt4_zh \
#    --template chatglm3 \
#    --output_dir data/alpaca_gpt4_zh/ \
#    --per_device_eval_batch_size 16 \
#    --predict_with_generate \

#accelerate launch src/train_bash.py \
#    --stage sft \
#    --model_name_or_path THUDM/chatglm3-6b \
#    --do_predict \
#    --dataset sharegpt_zh \
#    --template chatglm3 \
#    --output_dir data/sharegpt_zh/ \
#    --per_device_eval_batch_size 16 \
#    --predict_with_generate \
##    --finetuning_type lora \
##    --checkpoint_dir path_to_checkpoint \