#!/bin/bash

# 使用 timeout 命令设置最长运行时间为3小时
timeout 3h CUDA_VISIBLE_DEVICES=0,1,2,3 nohup accelerate launch \
    --config_file examples/accelerate/single_config.yaml \
    src/train.py examples/train/0515_chatglm/exp1.yaml > log/chatglm_exp1_0515.log 2>&1 &

# 等待第一个命令完成或超时
wait $!

# 运行第二个训练命令
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup accelerate launch \
    --config_file examples/accelerate/single_config.yaml \
    src/train.py examples/train/0513/exp1.yaml > log/qwen_exp1_0515.log 2>&1
