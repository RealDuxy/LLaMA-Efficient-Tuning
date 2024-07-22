USE_MODELSCOPE_HUB=1 CUDA_VISIBLE_DEVICES=0 nohup python \
    src/train.py examples/train/0620/qwen15-sft-exp3-predict.yaml > log/qwen15-sft-exp3-predict.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 python src/train.py examples/train/0620/qwen15-sft-exp3-predict.yaml