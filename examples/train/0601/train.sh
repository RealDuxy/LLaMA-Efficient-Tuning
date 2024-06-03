USE_MODELSCOPE_HUB=1 CUDA_VISIBLE_DEVICES=0 nohup python \
    src/train.py examples/train/0601/exp1_dpo.yaml > log/0601_qwen_rag_dpo_exp1.log 2>&1

