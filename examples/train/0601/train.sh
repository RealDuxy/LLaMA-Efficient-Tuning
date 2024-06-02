#USE_MODELSCOPE_HUB=1 CUDA_VISIBLE_DEVICES=0 nohup python \
#    src/train.py examples/train/0601/exp1_dpo.yaml > log/0601_qwen_rag_dpo_exp1.log 2>&1 &

USE_MODELSCOPE_HUB=1 CUDA_VISIBLE_DEVICES=0 nohup python \
    src/train.py examples/train/0601/exp1_simpo.yaml > log/0601_qwen_rag_simpo_exp1.log 2>&1 &

USE_MODELSCOPE_HUB=1 CUDA_VISIBLE_DEVICES=0 nohup python \
    src/train.py examples/train/0601/exp2_simpo.yaml > log/0601_qwen_rag_simpo_exp2.log 2>&1 &

USE_MODELSCOPE_HUB=1 CUDA_VISIBLE_DEVICES=0 nohup python \
    src/train.py examples/train/0601/exp3_simpo.yaml > log/0601_qwen_rag_simpo_exp3.log 2>&1 &


