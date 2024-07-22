#USE_MODELSCOPE_HUB=1 CUDA_VISIBLE_DEVICES=0 nohup python \
#    src/train.py examples/train/0620/qwen2-sft-exp1.yaml > log/qwen2-sft-exp1.log 2>&1

#USE_MODELSCOPE_HUB=1 CUDA_VISIBLE_DEVICES=0 python src/train.py examples/train/0620/qwen15-sft-exp4.yaml

## 睡眠半小时
#sleep 1h

USE_MODELSCOPE_HUB=1 CUDA_VISIBLE_DEVICES=0 nohup python \
    src/train.py examples/train/0620/qwen15-sft-exp3.yaml > log/qwen15-sft-exp3.log 2>&1

USE_MODELSCOPE_HUB=1 CUDA_VISIBLE_DEVICES=0 nohup python \
    src/train.py examples/train/0620/qwen15-sft-exp4.yaml > log/qwen15-sft-exp4.log 2>&1