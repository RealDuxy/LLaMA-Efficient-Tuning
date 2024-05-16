#!/bin/bash

# 运行第一个训练命令
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup accelerate launch \
    --config_file examples/accelerate/single_config.yaml \
    src/train.py examples/train/0515_chatglm/exp1.yaml > log/chatglm_exp1_0515.log 2>&1

# 等待第一个命令完全结束
wait

# 运行第二个训练命令
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup accelerate launch \
    --config_file examples/accelerate/single_config.yaml \
    src/train.py examples/train/0513/exp1.yaml > log/qwen_exp1_0515.log 2>&1

# 等待第二个命令完全结束
wait
