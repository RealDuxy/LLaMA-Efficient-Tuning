# 休眠5小时后运行
sleep 18000
USE_MODELSCOPE_HUB=1 CUDA_VISIBLE_DEVICES=0 nohup accelerate launch \
src/train.py examples/train/0513/exp2_dpo.yaml > log/dpo_qwen_exp2_0515.log 2>&1 &
