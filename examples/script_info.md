# 简介

## 命名格式
时间-（分布式）-训练/预测-任务名-实验轮次

## 记录
### 20240320
- 0320-train-askbob_qa_sft-exp1.sh
  - 模型：qwen1.5-7B/14B GPTQ 8/4 bits
  - 任务：askbob_qa的sft训练
  - 数据：askbob_qa 
  - 配置：v100
  
- 0320-ds-train-askbob_qa_comparison_dpo-exp1.sh
  - 模型：chatglm3
  - 任务：askbob_qa_comparison_0320 dpo训练
  - 数据：0205_stage1生成output, 由gpt4排序，共4k
  - 配置：4 * 32GB（V100）

### 20240313
- 0313-train-step_query_sft-exp1.sh
  - 模型：chatglm3
  - 任务：step_query的sft训练
  - 数据：gpt_4生成的stepback query数据1k 
  - 配置：4090
  
- 0313-train-rephrase_query_sft-exp1.sh
  - 模型：chatglm3
  - 任务：rephrase_query的sft训练
  - 数据：gpt_4生成的rephrase query数据1k 
  - 配置：4090 
  
- 0313-train-stepback_rephrase_query_sft-exp1.sh
  - 模型：chatglm3
  - 任务：rephrase_query和stepback_query的sft训练
  - 数据：gpt_4生成的rephrase query和stepback query数据各1k 
  - 配置：4090
  
- 0313-ds-train-askbob_qa_comparison_dpo-exp1.sh
  - 模型：chatglm3
  - 任务：askbob_qa_comparison dpo训练
  - 数据：0205_stage1生成output, 由gpt4排序，共6k
  - 配置：4 * 32GB（V100）


    