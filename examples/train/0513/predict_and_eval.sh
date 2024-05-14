# 并行执行 task1 和 task2
CUDA_VISIBLE_DEVICES=0 python ../src/train.py example/train/0513/predict_exp1.yaml > predict_exp1.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python ../src/train.py example/train/0513/predict_exp2.yaml > predict_exp2.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python ../src/train.py example/train/0513/predict_exp3.yaml > predict_exp3.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python ../src/train.py example/train/0513/predict_exp4.yaml > predict_exp4.log 2>&1 &