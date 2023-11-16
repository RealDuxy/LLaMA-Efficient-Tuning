#!/bin/bash


CUDA_VISIBLE_DEVICES=0 python ../src/train_bash.py \
    --stage sft \
    --do_train \
    --template chatglm3 \
    --dataset high_quality_qa,normal_qa,alpaca_gpt4_zh \
    --dataset_dir ../data \
    --finetuning_type lora \
    --output_dir ../checkpoints/1116_local_run \
    --max_samples 30000 \
    --overwrite_cache \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --lr_scheduler_type cosine \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_target all \
    --logging_steps 10 \
    --save_steps 500 \
    --learning_rate 5e-5 \
    --num_train_epochs 1.0 \
    --plot_loss \
    --cutoff_len 2500  \
    --reserved_label_len 800  \
    --fp16  \
    --neft_alpha 5 \
    --model_name_or_path THUDM/chatglm3-6b \
    #    --quantization_bit 8 \
#    --lora_target query_key_value,dense \
#!/bin/bash


