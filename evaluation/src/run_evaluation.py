# -*- encoding: utf-8 -*-
"""
@File    : run_evaluation.py
@Time    : 18/5/2024 00:13
@Author  : Duxy
@Email   : du.xi.yang@qq.com
@Software: PyCharm
"""


import json

import jieba
from tqdm import tqdm
from transformers import AutoTokenizer
from rouge_chinese import Rouge

# Load the tokenizer and model for the specified transformer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-14B-Chat", trust_remote_code=True)

# Function to compute ROUGE scores
import json
import jieba
from rouge import Rouge
import math


# 计算两个文本之间的ROUGE分数
def compute_rouge_scores(prediction, reference):
    hypothesis = list(jieba.cut(prediction))
    reference = list(jieba.cut(reference))

    if len(" ".join(hypothesis).split()) == 0 or len(" ".join(reference).split()) == 0:
        result = 0.0
    else:
        rouge = Rouge()
        scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
        result = scores[0]['rouge-1']['f']  # 使用ROUGE-1 F分数作为结果

    return result


# 加载数据
def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


# 描述分数的分布
def describe(data, percentiles=[25, 50, 75, 90]):
    count = len(data)
    mean = sum(data) / count
    variance = sum((x - mean) ** 2 for x in data) / count
    std_dev = math.sqrt(variance)
    min_val = min(data)
    max_val = max(data)

    percentiles_data = {}
    sorted_data = sorted(data)
    for p in percentiles:
        if 0 < p < 100:
            k = f"{p}%"
            index = int(math.ceil((len(sorted_data) * p) / 100)) - 1
            percentiles_data[k] = sorted_data[index]

    description = {
        'count': count,
        'mean': round(mean, 2),
        'min': min_val,
        'percentiles': percentiles_data,
        'max': max_val
    }

    return description


# 主程序
def main(filepath, output_file):
    data = load_data(filepath)

    # 计算分数
    scores = []
    for item in tqdm(data):
        score = compute_rouge_scores(item['pred'], item['output'])
        if score >= 0.95:
            print(f"output: \n {item['output']}")
            print(f"pred: \n {item['pred']}")
        scores.append(score)

    # 输出分数分布
    score_description = describe(scores)
    print("分数分布:", score_description)

    # 选取最小的30%的数据
    threshold = sorted(scores)[int(len(scores) * 0.3)]
    print(f"threshold: {threshold}")
    func = lambda x: {"question": x["question"],
                   "contexts": x["context"],
                   "output": [x["output"], x["pred"]]}
    selected_data = [func(data[i]) for i, score in enumerate(scores) if score <= threshold]

    # 保存选中的数据
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(selected_data, file, ensure_ascii=False, indent=4)

    print(f"已保存最小30%分数的数据，共{len(selected_data)}条。")



if __name__ == '__main__':
    filepath = "output/train_dataset/chatglm-rag-0515/train_dynamic_cot_trigger_output.json"
    output_file = "output/train_dataset/chatglm-rag-0515/train_dynamic_cot_trigger_comparison.json"
    main(filepath, output_file)

    filepath = "output/train_dataset/chatglm-rag-0515/train_instruction_only_output.json"
    output_file = "output/train_dataset/chatglm-rag-0515/train_instruction_only_comparison.json"
    main(filepath, output_file)

    filepath = "output/train_dataset/chatglm-rag-0515/train_fix_cot_trigger_output.json"
    output_file = "output/train_dataset/chatglm-rag-0515/train_fix_cot_trigger_comparison.json"
    main(filepath, output_file)