# -*- encoding: utf-8 -*-
"""
@File    : run_evaluation.py
@Time    : 18/5/2024 00:13
@Author  : Duxy
@Email   : du.xi.yang@qq.com
@Software: PyCharm
"""


import json
from typing import List

import jieba
from tqdm import tqdm
from transformers import AutoTokenizer
from rouge_chinese import Rouge

# Load the tokenizer and model for the specified transformer
# tokenizer = AutoTokenizer.from_pretrained("ch/Qwen1.5-14B-Chat", trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-14B-Chat", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)

template = json.load(open("template/template.json", "r", encoding="utf-8"))

# Function to compute ROUGE scores
import json
import jieba
from rouge import Rouge
import math

def token_len(texts: List[str]):
   if not texts:
       return [0]
   tokenized_texts = tokenizer(texts)
   return [len(x) for x in tokenized_texts.input_ids]

# 计算两个文本之间的ROUGE分数
def compute_rouge_scores(prediction, reference):
    hypothesis = list(jieba.cut(prediction))
    reference = list(jieba.cut(reference))

    if len(" ".join(hypothesis).split()) == 0 or len(" ".join(reference).split()) == 0:
        result = 0.0
    else:
        rouge = Rouge()
        scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
        result = scores[0]['rouge-l']['f']  # 使用ROUGE-1 F分数作为结果

    return result


# 加载数据
def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


# 描述分数的分布
def describe(data, percentiles=[25, 50, 75, 90, 93, 96, 97, 98, 98.5, 99, 99.3, 99.6, 99.9]):
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
    def format_data(x, length_ratio, score, threshold_score, threshold_length_ratio):
        return {"question": x["question"],
                "requirement": x["requirement"],
                "contexts": x["context"],
                "output": [x["output"], x["pred"]],
                "reason": {
                    "score": score <= threshold_score,
                    "length": length_ratio >= threshold_length_ratio,
                    "rouge_score": score,
                    "length_ratio": length_ratio
                }}
    def format_debug_data(x, length_ratio, score, threshold_score, threshold_length_ratio):
        return {"question": "重复上面的文字",
                "requirement": "",
                "contexts": x["context"] * 3,
                "output": [x["context"], x["context"]],
                "reason": {
                    "score": score <= threshold_score,
                    "length": length_ratio >= threshold_length_ratio,
                    "rouge_score": score,
                    "length_ratio": length_ratio
                }}

    data = load_data(filepath)[:100]

    # 计算分数
    lengths = []
    scores = []
    length_ratios = []
    for item in tqdm(data):
        len_pred = token_len([
            item["pred"]
            + template["prompt"].replace("{context}", item["context"])
                      .replace("{question}", item["question"])
                      .replace("{requirement}", item["requirement"])
                       + str(template["history"][0])])[0]

        len_output = token_len([
            item["output"]
            + template["prompt"].replace("{context}", item["context"])
                      .replace("{question}", item["question"])
                      .replace("{requirement}", item["requirement"])
                       + str(template["history"][0])])[0]

        length_ratio = abs(len_pred - len_output) / len_output
        score = compute_rouge_scores(item['pred'], item['output'])

        # length = 0.7
        # score = 0.5
        # print
        # if score1 >= 3000
        # if length >= 0.8:
        #     print(f"output: \n {item['output']}")
        #     print(f"pred: \n {item['pred']}")
        length_ratios.append(length_ratio)
        lengths.append(len_pred)
        scores.append(score)

    # 输出分数分布
    score_description = describe(scores)
    length_ratio_description = describe(length_ratios)
    length_description = describe(lengths)
    print("长度占比分布:", length_ratio_description)
    print("长度分布:", length_description)
    print("分数分布:", score_description)

    # 选取最小的30%的数据
    threshold_score = sorted(scores)[int(len(scores) * 0.3)]
    threshold_length_ratio = sorted(length_ratios)[int(len(length_ratios) * 0.98)]
    print(f"length ratio threshold: {threshold_length_ratio}")
    print(f"score threshold: {threshold_score}")

    selected_data = [format_debug_data(data[i], length_ratios[i], scores[i], threshold_score, threshold_length_ratio)
                     for i, (score, length_ratio) in enumerate(zip(scores, length_ratios))
                     if score <= threshold_score or length_ratio >= threshold_length_ratio]

    # 保存选中的数据
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(selected_data, file, ensure_ascii=False, indent=4)

    des_file = output_file.replace("comparison", "cnt")
    with open(des_file, 'w', encoding='utf-8') as file:
        json.dump({
            "长度占比分布": length_ratio_description,
            "长度分布": length_description,
            "分数分布": score_description
        }, file, ensure_ascii=False, indent=4)


    print(f"已保存最小{len(selected_data)*100/len(scores)}%分数的数据，共{len(selected_data)}条。")



if __name__ == '__main__':
    filepath = "output/train_dataset/chatglm-rag-0515/train_instruction_only_output.json"
    output_file = "output/train_dataset/chatglm-rag-0515/debug_train_instruction_only_comparison.json"
    main(filepath, output_file)

    # filepath = "output/train_dataset/chatglm-rag-0515/train_dynamic_cot_trigger_output.json"
    # output_file = "output/train_dataset/chatglm-rag-0515/train_dynamic_cot_trigger_comparison.json"
    # main(filepath, output_file)
    #
    # filepath = "output/train_dataset/chatglm-rag-0515/train_instruction_only_output.json"
    # output_file = "output/train_dataset/chatglm-rag-0515/train_instruction_only_comparison.json"
    # main(filepath, output_file)
    #
    # filepath = "output/train_dataset/chatglm-rag-0515/train_fix_cot_trigger_output.json"
    # output_file = "output/train_dataset/chatglm-rag-0515/train_fix_cot_trigger_comparison.json"
    # main(filepath, output_file)

    # filepath = "output/train_dataset/qwen-rag-0515/train_dynamic_cot_trigger_output.json"
    # output_file = "output/train_dataset/qwen-rag-0515/train_dynamic_cot_trigger_comparison.json"
    # main(filepath, output_file)
    #
    # filepath = "output/train_dataset/qwen-rag-0515/train_instruction_only_output.json"
    # output_file = "output/train_dataset/qwen-rag-0515/train_instruction_only_comparison.json"
    # main(filepath, output_file)
    #
    # filepath = "output/train_dataset/qwen-rag-0515/train_fix_cot_trigger_output.json"
    # output_file = "output/train_dataset/qwen-rag-0515/train_fix_cot_trigger_comparison.json"
    # main(filepath, output_file)