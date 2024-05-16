#!/bin/bash

#NPROC_PER_NODE=4
#
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
#    --nproc_per_node $NPROC_PER_NODE \
#    --nnodes 1 \
#    --standalone \
#    src/train.py examples/train/0515_chatglm/exp1.yaml
#
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
#    --nproc_per_node 4 \
#    --nnodes 1 \
#    --standalone \
#    src/train.py examples/train/0515_chatglm/exp1.yaml
#
#CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
#    --config_file examples/accelerate/single_config.yaml \
#    src/train.py examples/train/0515_chatglm/exp1.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup accelerate launch \
    --config_file examples/accelerate/single_config.yaml \
    src/train.py examples/train/0515_chatglm/exp1.yaml > log/chatglm_exp1_0515.log 2>&1

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup accelerate launch \
    --config_file examples/accelerate/single_config.yaml \
    src/train.py examples/train/0513/exp1.yaml > log/qwen_exp1_0515.log 2>&1