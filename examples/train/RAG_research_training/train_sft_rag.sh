CUDA_VISIBLE_DEVICES=0 nohup python src/train.py examples/train/RAG_research_training/train_nq_seq.yaml > log/0805-llama3-8b-lora-alpaca-nq-seq-exp1.log 2>&1

CUDA_VISIBLE_DEVICES=0 nohup python src/train.py examples/train/RAG_research_training/train_nq_mix.yaml > log/0805-llama3-8b-lora-alpaca-nq-mix-exp1.log 2>&1

## no nohup
#CUDA_VISIBLE_DEVICES=0 python src/train.py examples/train/RAG_research_training/train_boolq.yaml
#CUDA_VISIBLE_DEVICES=0 python src/train.py examples/train/RAG_research_training/train_obqa_fs.yaml
#CUDA_VISIBLE_DEVICES=0 python src/train.py examples/train/RAG_research_training/train_obqa_zs.yaml

