## 并行执行 task1 和 task2
#USE_MODELSCOPE_HUB=1 CUDA_VISIBLE_DEVICES=0 python src/train.py examples/train/0513/exp1.yaml > log/exp1.log 2>&1 &
#sleep 10
#USE_MODELSCOPE_HUB=1 CUDA_VISIBLE_DEVICES=1 python src/train.py examples/train/0513/exp2.yaml > log/exp2.log 2>&1 &
#sleep 10
## 在后台执行串行任务的脚本
#(
#  USE_MODELSCOPE_HUB=1CUDA_VISIBLE_DEVICES=2 python src/train.py examples/train/0513/exp3.yaml > log/exp3.log 2>&1
#  USE_MODELSCOPE_HUB=1CUDA_VISIBLE_DEVICES=2 python src/train.py examples/train/0513/exp4.yaml > log/exp4.log 2>&1
#) &
echo "estimate gpu memory = 32413mb"
echo "Start training exp2"
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/train.py examples/train/0513/exp2.yaml > log/exp2.log 2>&1
echo "Start training exp3"
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/train.py examples/train/0513/exp3.yaml > log/exp3.log 2>&1
echo "Start training exp4"
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/train.py examples/train/0513/exp4.yaml > log/exp4.log 2>&1
echo "Start training exp1"
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate run src/train.py examples/train/0513/exp1.yaml > log/exp1_0515.log 2>&1 &
echo "training complete"


CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --config_file examples/accelerate/single_config.yaml \
    src/train.py examples/train/0513/exp1.yaml