


USE_MODELSCOPE_HUB=1 CUDA_VISIBLE_DEVICES=0 nohup python \
src/train.py examples/train/0513/exp2.yaml > log/sft_qwen_exp2_0527.log 2>&1

USE_MODELSCOPE_HUB=1 CUDA_VISIBLE_DEVICES=0 nohup python \
    src/train.py examples/train/0513/exp2_simpo.yaml > log/simpo_qwen_exp2_0529.log 2>&1

#USE_MODELSCOPE_HUB=1 CUDA_VISIBLE_DEVICES=0 nohup accelerate launch \
#    src/train.py examples/train/0513/exp1_dpo.yaml > log/dpo_qwen_exp1_0515.log 2>&1 &
#
#USE_MODELSCOPE_HUB=1 CUDA_VISIBLE_DEVICES=0 nohup accelerate launch \
#src/train.py examples/train/0513/exp2_dpo.yaml > log/dpo_qwen_exp2_0515.log 2>&1 &
#
#USE_MODELSCOPE_HUB=1 CUDA_VISIBLE_DEVICES=0 nohup accelerate launch \
#    --config_file examples/accelerate/single_config.yaml \
#    src/train.py examples/train/0513/exp1_dpo.yaml > log/dpo_qwen_exp1_0515.log 2>&1 &


