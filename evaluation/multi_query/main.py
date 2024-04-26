# -*- encoding: utf-8 -*-
"""
@File    : rag.py
@Time    : 27/2/2024 16:25
@Author  : Duxy
@Email   : du.xi.yang@qq.com
@Software: PyCharm
"""
import json
import re

import pandas as pd
from tqdm import tqdm

from evaluation.multi_query.base import BaseAgent
from evaluation.multi_query.local_model_infer import get_chatglm_response, get_qwen_response


def correct_list_str(list_str):
    "修正可能错误的json string"
    list_str = list_str.replace('\'', '"')
    list_str = list_str.replace('“', '"').replace("”", '"')
    list_str = list_str.replace('‘', '"').replace("’", '"')
    list_str = list_str.replace("，", ',').replace("：", ':')
    list_str = re.sub(r'\]\]+', ']', list_str)
    list_str = re.sub(r'\[\[+', '[', list_str)
    list_str = re.sub(r'\,\,+', ',', list_str)
    list_str = re.sub(r'\:\:+', ':', list_str)
    return list_str


def extract_and_correct_list(text):
    "提取答案中的list"
    match = re.search(r'\[.*?\]', text)
    if match:
        list_str = match.group()
        list_str = correct_list_str(list_str)
        try:
            list_str_data = eval(list_str)
            if isinstance(list_str_data, list):
                map(str, list_str_data)
            return list_str_data
        except Exception:
            return []
    else:
        return []

def run_test_for_multi_query_generation(data_path,
                                        post_fix,
                                        template_file,
                                        model_path,
                                        peft_path,
                                        model_invoke=get_qwen_response):

    multiquery_agent = BaseAgent(template_file=template_file, model_invoke=model_invoke, model_path=model_path, peft_path=peft_path)
    df = pd.read_excel(data_path).to_dict("records")
    for i, line in tqdm(enumerate(df)):
        question = line["输入"]
        if isinstance(question, str):
            output = multiquery_agent.invoke(query=question)
            response = extract_and_correct_list(output)
            if response == []:
                print(f"current question: {question}")
                print(f"current output: {output}")
                print()
            df[i]["输出"] = response
        else:
            print(f"current line: {line}")
            df[i]["输出"] = []
            df[i]["输出"] = response
            continue

    pd.DataFrame(df).to_excel(f"评估集0306_{post_fix}_output.xlsx", index=False)

if __name__ == '__main__':
    # qwen的rephrase
    # # vanilla
    model_path = "/mnt/d/PycharmProjects/models/Qwen1.5-14B-Chat-GPTQ-Int4/"
    # peft_path = None
    # run_test_for_multi_query_generation(data_path="评估集0306.xlsx", post_fix="qwen_rephrase_query_generation_vanilla",
    #                                     template_file="./template/rephrase.json", model_path=model_path, peft_path=peft_path)
    # # mle训练
    # peft_path = "../../checkpoints/0423_rephrase_query_qwen_exp1"
    # run_test_for_multi_query_generation(data_path="评估集0306.xlsx", post_fix="qwen_rephrase_query_generation_mle",
    #                                     template_file="./template/rephrase_0424.json")
    # emo训练
    # peft_path = "../../checkpoints/0423_rephrase_query_qwen_exp2"
    # run_test_for_multi_query_generation(data_path="评估集0306.xlsx", post_fix="qwen_rephrase_query_generation_emo",
    #                                     template_file="./template/rephrase_0424.json", model_path=model_path, peft_path=peft_path)

    # qwen的stepback
    # vanilla
    # peft_path = None
    # run_test_for_multi_query_generation(data_path="评估集0306.xlsx", post_fix="qwen_stepback_query_generation_vanilla",
    #                                     template_file="./template/stepback.json", model_path=model_path, peft_path=peft_path)
    # # mle训练
    # peft_path = "../../checkpoints/0423_stepback_query_qwen_exp1"
    # run_test_for_multi_query_generation(data_path="评估集0306.xlsx", post_fix="qwen_stepback_query_generation_mle",
    #                                     template_file="./template/stepback_0424.json", model_path=model_path, peft_path=peft_path)
    # emo训练
    peft_path = "../../checkpoints/0423_stepback_query_qwen_exp2"
    run_test_for_multi_query_generation(data_path="评估集0306.xlsx", post_fix="qwen_stepback_query_generation_emo",
                                        template_file="./template/stepback_0424.json", model_path=model_path, peft_path=peft_path)