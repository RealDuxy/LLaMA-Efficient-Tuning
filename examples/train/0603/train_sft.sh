USE_MODELSCOPE_HUB=1 CUDA_VISIBLE_DEVICES=0,1,2,3 nohup accelerate launch src/train.py examples/train/0603-qwen-full/exp1.yaml > log/0603_qwen-7b-rag-exp1.log 2>&1

USE_MODELSCOPE_HUB=1 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun src/train.py examples/train/0603-qwen-full/exp1.yaml

USE_MODELSCOPE_HUB=1 CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --num_gpus=2 src/train.py examples/train/0603-qwen-full/exp1.yaml

USE_MODELSCOPE_HUB=1 CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train examples/train/0603-qwen-full/exp1.yaml