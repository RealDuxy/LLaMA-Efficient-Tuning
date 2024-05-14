# 并行执行 task1 和 task2
CUDA_VISIBLE_DEVICES=0 python ../src/train.py ../example/train/0513/exp1.yaml > ../log/exp1.log 2>&1 &
sleep 30
CUDA_VISIBLE_DEVICES=1 python ../src/train.py ../example/train/0513/exp2.yaml > ../log/exp2.log 2>&1 &
sleep 30
# 在后台执行串行任务的脚本
(
  CUDA_VISIBLE_DEVICES=2 python ../src/train.py ../example/train/0513/exp3.yaml > ../log/exp3.log 2>&1
  CUDA_VISIBLE_DEVICES=2 python ../src/train.py ../example/train/0513/exp4.yaml > ../log/exp4.log 2>&1
) &