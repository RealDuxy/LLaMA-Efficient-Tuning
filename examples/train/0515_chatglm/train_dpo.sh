#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
  --config_file examples/accelerate/single_config.yaml \
  src/train.py examples/train/0515_chatglm/exp2_dpo.yaml > log/dpo_chatglm_exp2_0519.log 2>&1 &
