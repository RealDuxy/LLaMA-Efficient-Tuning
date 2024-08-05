#USE_MODELSCOPE_HUB=1 CUDA_VISIBLE_DEVICES=0 nohup python \
#    src/train.py examples/train/0620/qwen2-sft-exp1.yaml > log/qwen2-sft-exp1.log 2>&1

#USE_MODELSCOPE_HUB=1 CUDA_VISIBLE_DEVICES=0 python src/train.py examples/train/0620/qwen15-sft-exp4.yaml

## 睡眠半小时
#sleep 1h


#CUDA_VISIBLE_DEVICES=0 python src/train.py examples/train/RAG_research_training/train_alpaca_instruct.yaml
#
#CUDA_VISIBLE_DEVICES=0 python src/train.py examples/train/RAG_research_training/train_nq_mix.yaml
#
#CUDA_VISIBLE_DEVICES=0 python src/train.py examples/train/RAG_research_training/train_nq_seq.yaml

CUDA_VISIBLE_DEVICES=0 nohup python src/train.py examples/train/RAG_research_training/train_alpaca_instruct.yaml > log/llama3-alpaca-lora.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python src/train.py examples/train/RAG_research_training/train_rag_sft.yaml > log/0729-llama3-8b-lora-alpaca-nq-seq.log 2>&1

CUDA_VISIBLE_DEVICES=0 nohup python src/train.py examples/train/RAG_research_training/train_rag.yaml > log/0729-llama3-8b-lora-alpaca-nq-mix.log 2>&1

CUDA_VISIBLE_DEVICES=0 nohup python src/train.py examples/train/RAG_research_training/train_boolq.yaml > log/0805-llama3-8b-lora-boolq-exp1.log 2>&1

CUDA_VISIBLE_DEVICES=0 nohup python src/train.py examples/train/RAG_research_training/train_obqa_fs.yaml > log/0805-llama3-8b-lora-obqa-fs-exp1.log 2>&1

CUDA_VISIBLE_DEVICES=0 nohup python src/train.py examples/train/RAG_research_training/train_obqa_zs.yaml > log/0805-llama3-8b-lora-obqa-zs-exp1.log 2>&1

# no nohup
CUDA_VISIBLE_DEVICES=0 python src/train.py examples/train/RAG_research_training/train_boolq.yaml
CUDA_VISIBLE_DEVICES=0 python src/train.py examples/train/RAG_research_training/train_obqa_fs.yaml
CUDA_VISIBLE_DEVICES=0 python src/train.py examples/train/RAG_research_training/train_obqa_zs.yaml

