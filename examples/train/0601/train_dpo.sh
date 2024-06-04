#USE_MODELSCOPE_HUB=1 CUDA_VISIBLE_DEVICES=0 nohup python \
#    src/train.py examples/train/0601/exp1_dpo.yaml > log/0601_qwen_rag_dpo_exp1.log 2>&1

#USE_MODELSCOPE_HUB=1 CUDA_VISIBLE_DEVICES=0 nohup python \
#    src/train.py examples/train/0601/exp2_dpo.yaml > log/0601_qwen_rag_dpo_exp2.log 2>&1

#USE_MODELSCOPE_HUB=1 CUDA_VISIBLE_DEVICES=0 nohup python \
#    src/train.py examples/train/0601/exp6_dpo.yaml > log/0601_qwen_rag_dpo_exp6.log 2>&1

USE_MODELSCOPE_HUB=1 CUDA_VISIBLE_DEVICES=0 nohup python \
    src/train.py examples/train/0601/exp5_dpo.yaml > log/0601_qwen_rag_dpo_exp5.log 2>&1

#USE_MODELSCOPE_HUB=1 CUDA_VISIBLE_DEVICES=0 nohup python \
#    src/train.py examples/train/0601/exp4_dpo.yaml > log/0601_qwen_rag_dpo_exp4.log 2>&1

#USE_MODELSCOPE_HUB=1 CUDA_VISIBLE_DEVICES=0 nohup python \
#    src/train.py examples/train/0601/exp3_dpo.yaml > log/0601_qwen_rag_dpo_exp3.log 2>&1
