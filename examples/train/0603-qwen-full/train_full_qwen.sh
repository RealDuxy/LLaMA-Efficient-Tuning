USE_MODELSCOPE_HUB=1 CUDA_VISIBLE_DEVICES=0 nohup python \
    src/train.py examples/train/0603-qwen-full/exp1.yaml > log/0603_qwen-7b-rag-exp1.log 2>&1
