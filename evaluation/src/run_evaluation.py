# -*- encoding: utf-8 -*-
"""
@File    : run_evaluation.py
@Time    : 18/5/2024 00:13
@Author  : Duxy
@Email   : du.xi.yang@qq.com
@Software: PyCharm
"""


import json
import time
from typing import List

import jieba
from tqdm import tqdm
from transformers import AutoTokenizer

from evaluation.src.utils import Rouge

# Load the tokenizer and model for the specified transformer
tokenizer = AutoTokenizer.from_pretrained("/mnt/d/PycharmProjects/models/Qwen1.5-14B-Chat-GPTQ-Int4", trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-14B-Chat", trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)

template = json.load(open("template/template.json", "r", encoding="utf-8"))

# Function to compute ROUGE scores
import json
import jieba
import math

def token_len(texts: List[str]):
   if not texts:
       return [0]
   tokenized_texts = tokenizer(texts)
   return [len(x) for x in tokenized_texts.input_ids]

def tokenize_text(texts: List[str]):
    tokenized_ids = tokenizer(texts).input_ids
    # response = tokenizer.batch_decode(tokenized_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    result = []
    for ids in tokenized_ids:
        tmp = [str(id) for id in ids]
        result.append(tmp)
    return result

# 计算两个文本之间的ROUGE分数
def compute_rouge_scores(predictions, references):

    time_1 = time.time()
    # hypothesises = [" ".join(jieba.lcut(prediction)) for prediction in predictions]
    # references = [" ".join(jieba.lcut(reference)) for reference in references]
    hypothesises = [" ".join(prediction) if predictions else "x" for prediction in tokenize_text(predictions)]
    references = [" ".join(reference) if references else "x" for reference in tokenize_text(references)]
    time_2 = time.time()
    # if len(" ".join(hypothesis).split()) == 0 or len(" ".join(reference).split()) == 0:
    #     result = 0.0
    # else:
    #     rouge = Rouge()
    #     scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
    #     result = scores[0]['rouge-l']['f']  # 使用ROUGE-1 F分数作为结果

    rouge = Rouge()
    scores = rouge.get_scores(hypothesises, references)
    result = [x['rouge-l']['f'] for x in scores]
    time_3 = time.time()
    # 耗时打印
    print(f"time cost for tokenize: {time_2-time_1}")
    print(f"time cost for compute rouge: {time_3-time_2}")
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
def main(filepath, output_file, percents):
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

    data = load_data(filepath)

    # 计算分数
    length_pred_list = token_len([item["pred"] for item in tqdm(data, desc="tokenizing predictions")])
    length_label_list = token_len([item["output"] for item in tqdm(data, desc="tokenizing labels")])
    length_ratios = [(len_output - len_pred) / len_output for len_pred, len_output in zip(length_pred_list, length_label_list)]
    scores = compute_rouge_scores([item['pred'] for item in data],
                                   [item['output'] for item in data])
    # for item in tqdm(data):
    #     pred = item["pred"]
    #     output = item["output"]
    #     length_both = token_len([pred, output])
    #     len_pred = length_both[0]
    #     len_output = length_both[1]

    ## 输出分数
    #     len_pred = token_len([
    #         item["pred"]
    #         + template["prompt"].replace("{context}", item["context"])
    #                   .replace("{question}", item["question"])
    #                   .replace("{requirement}", item["requirement"])
    #                    + str(template["history"][0])])[0]
    #
    #     len_output = token_len([
    #         item["output"]
    #         + template["prompt"].replace("{context}", item["context"])
    #                   .replace("{question}", item["question"])
    #                   .replace("{requirement}", item["requirement"])
    #                    + str(template["history"][0])])[0]

        # length_ratio = abs(len_pred - len_output) / len_output
        # score = compute_rouge_scores(item['pred'], item['output'])

        # length = 0.7
        # score = 0.5
        # print
        # if score1 >= 3000
        # if length_ratio >= 0.5:
        #     print(f"output: \n {item['output']}")
        #     print(f"pred: \n {item['pred']}")
        # length_ratios.append(length_ratio)
        # lengths.append(len_pred)
        # scores.append(score)

    # 输出分数分布
    score_description = describe(scores)
    length_ratio_description = describe(length_ratios)
    length_pred_description = describe(length_pred_list)
    length_label_description = describe(length_label_list)
    print("长度占比分布:", length_ratio_description)
    print("预测长度分布:", length_pred_description)
    print("目标长度分布:", length_label_description)
    print("分数分布:", score_description)

    # 选取最小的30%的数据
    for percent in percents:
        threshold_score = sorted(scores)[int(len(scores) * (1-percent))]
        threshold_length_ratio = sorted(length_ratios)[int(len(length_ratios) * (1-percent))]
        print(f"length ratio threshold: {threshold_length_ratio}")
        print(f"score threshold: {threshold_score}")

        selected_data = [format_data(data[i], length_ratios[i], scores[i], threshold_score, threshold_length_ratio)
                         for i, (score, length_ratio) in enumerate(zip(scores, length_ratios))
                         if score <= threshold_score or length_ratio >= threshold_length_ratio]

        # 保存选中的数据
        with open(output_file.replace(".json", f"_{percent*100}p.json"), 'w', encoding='utf-8') as file:
            json.dump(selected_data, file, ensure_ascii=False, indent=4)

        des_file = output_file.replace("comparison", "cnt")
        with open(des_file, 'w', encoding='utf-8') as file:
            json.dump({
                "长度占比分布": length_ratio_description,
                "预测长度分布": length_pred_description,
                "目标长度分布": length_label_description,
                "分数分布": score_description
            }, file, ensure_ascii=False, indent=4)


        print(f"已保存最小{len(selected_data)*100/len(scores)}%分数的数据，共{len(selected_data)}条。")



if __name__ == '__main__':
    # filepath = "output/train_dataset/chatglm-rag-0515/train_instruction_only_output.json"
    # output_file = "output/train_dataset/chatglm-rag-0515/debug_train_instruction_only_comparison.json"
    # main(filepath, output_file)

    filepath = "output/train_dataset/qwen-rag-0529-exp2/train_0524_dynamic_cot_trigger_output.json"
    output_file = "output/train_dataset/qwen-rag-0529-exp2/train_0524_dynamic_cot_trigger_comparison.json"
    main(filepath, output_file, percents=[1.0, 0.25, 0.1])

    filepath = "output/train_dataset/qwen-rag-0529-exp2/train_0524_instruction_only_output.json"
    output_file = "output/train_dataset/qwen-rag-0529-exp2/train_0524_instruction_only_comparison.json"
    main(filepath, output_file, percents=[1.0, 0.25, 0.1])

    filepath = "output/train_dataset/qwen-rag-0529-exp2/train_0524_fix_cot_trigger_output.json"
    output_file = "output/train_dataset/qwen-rag-0529-exp2/train_0524_fix_cot_trigger_comparison.json"
    main(filepath, output_file, percents=[1.0, 0.25, 0.1])

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