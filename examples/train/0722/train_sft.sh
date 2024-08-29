#USE_MODELSCOPE_HUB=1 CUDA_VISIBLE_DEVICES=0 nohup python \
#    src/train.py examples/train/0620/qwen2-sft-exp1.yaml > log/qwen2-sft-exp1.log 2>&1

#USE_MODELSCOPE_HUB=1 CUDA_VISIBLE_DEVICES=0 python src/train.py examples/train/0620/qwen15-sft-exp4.yaml

## 睡眠半小时
#sleep 1h

USE_MODELSCOPE_HUB=1 CUDA_VISIBLE_DEVICES=0 nohup python \
    src/train.py examples/train/0620/qwen15-sft-exp3.yaml > log/qwen15-sft-exp3.log 2>&1

USE_MODELSCOPE_HUB=1 CUDA_VISIBLE_DEVICES=0 nohup python \
    src/train.py examples/train/0620/qwen15-sft-exp4.yaml > log/qwen15-sft-exp4.log 2>&1


USE_MODELSCOPE_HUB=1 CUDA_VISIBLE_DEVICES=0 nohup python src/train.py examples/train/0722/qwen15-stage1-exp1.yaml > log/qwen15-stage1-0722-exp1.log 2>&1 &

USE_MODELSCOPE_HUB=1 CUDA_VISIBLE_DEVICES=0 nohup python src/train.py examples/train/0722/qwen15-stage1-exp2.yaml > log/qwen15-stage1-0722-exp2.log 2>&1

CUDA_VISIBLE_DEVICES=0 nohup python src/train.py examples/train/0722/qwen15-stage2-exp1.yaml > log/qwen15-stage2-0815-exp1.log 2>&1 &


USE_MODELSCOPE_HUB=1 CUDA_VISIBLE_DEVICES=0 python src/train.py examples/train/0722/qwen15-stage1-exp1.yaml

USE_MODELSCOPE_HUB=1 CUDA_VISIBLE_DEVICES=0 python src/train.py examples/train/0722/qwen15-stage1-exp2.yaml

USE_MODELSCOPE_HUB=1 CUDA_VISIBLE_DEVICES=0 python src/train.py examples/train/0722/qwen15-stage2-exp1.yaml