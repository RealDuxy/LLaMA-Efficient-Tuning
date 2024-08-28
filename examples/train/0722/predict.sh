python src/train.py examples/train/0722/qwen15-sft-stage12-predict.yaml

CUDA_VISIBLE_DEVICES=0 nohup python src/train.py examples/train/0722/qwen15-sft-stage12-predict.yaml > log/qwen15-sft-stage2.log 2>&1


API_PORT=8000 llamafactory-cli api examples/inference/qwen15_vllm.yaml

python src/api.py examples/inference/qwen15_vllm.yaml