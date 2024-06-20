#!/bin/bash

# 使用 timeout 命令设置最长运行时间为3小时，同时正确设置环境变量
timeout 3h bash -c 'CUDA_VISIBLE_DEVICES=0,1,2,3 nohup accelerate launch \
    --config_file examples/accelerate/single_config.yaml \
    src/train.py examples/train/0515_chatglm/qwen2-sft-exp1.yaml > log/chatglm_exp1_0515.log 2>&1'

# 等待第一个命令完成或超时
wait

# 运行第二个训练命令
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup accelerate launch \
    --config_file examples/accelerate/single_config.yaml \
    src/train.py examples/train/0513/qwen2-sft-exp1.yaml > log/qwen_exp1_0515.log 2>&1
