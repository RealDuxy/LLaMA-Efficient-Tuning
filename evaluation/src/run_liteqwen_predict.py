# -*- encoding: utf-8 -*-
"""
@File    : rag.py
@Time    : 15/5/2024 12:25
@Author  : Duxy
@Email   : du.xi.yang@qq.com
@Software: PyCharm
"""

import json
import os.path
import re
import sys
from random import shuffle

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from base import BaseLiteLLMAgent
from utils import get_qwen_response, batch_dataset_iterator, get_chatglm_response

model_adapter_name_map = {
    "chatglm": "",
    "chatglm-rag-0515": "default",
    "chatglm-rag-0515-dpo": "align",
    "qwen": "",
    "qwen-rag-0515": "default"
}

def run_rag_evaluation(data_dir, output_dir,
                       template_file,
                       model_name="names",
                       max_samples=None,
                       model_invoke=get_qwen_response):
    rag_agent = BaseLiteLLMAgent(template_file=template_file, model_invoke=model_invoke)

    for data_file in os.listdir(data_dir):
        output_file = data_file.replace(".json", f"_output.json")
        data_file = os.path.join(data_dir, data_file)
        model_output_dir = output_dir + f"/{model_name}/"
        os.makedirs(model_output_dir, exist_ok=True)
        output_file = os.path.join(model_output_dir, output_file)
        print(f"Processing data file: {data_file}")
        results = []
        for datas in tqdm(batch_dataset_iterator(data_file, batch_size=4, max_samples=max_samples)):
            predictions = rag_agent.para_invoke(adapter_name=[model_name] * len(datas["question"]),
                                                **{"question": datas["question"]
                                                    , "requirement": datas["requirement"]
                                                    , "context": datas["context"]})
            datas.update({"pred": predictions})
            for i in range(len(predictions)):
                results.append({
                    "question": datas["question"][i],
                    "requirement": datas["requirement"][i],
                    "context": datas["context"][i],
                    "output": datas["output"][i],
                    "pred": datas["pred"][i]
                })

        # results 保存到output_file
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"Results saved to {output_file}")


def run_rag_all_negative_rejection_answer(data_dir, output_dir,
                                          template_file,
                                          model_name="names",
                                          max_samples=None,
                                          model_invoke=get_chatglm_response):
    rag_agent = BaseLiteLLMAgent(template_file=template_file, model_invoke=model_invoke)

    for data_file in os.listdir(data_dir):
        output_file = data_file.replace(".json", f"_output.json")
        data_file = os.path.join(data_dir, data_file)
        model_output_dir = output_dir + f"/{model_name}/"
        os.makedirs(model_output_dir, exist_ok=True)
        output_file = os.path.join(model_output_dir, output_file)
        print(f"Processing data file: {data_file}")
        results = []
        for datas in tqdm(batch_dataset_iterator(data_file, batch_size=4, max_samples=max_samples)):
            predictions = rag_agent.para_invoke(adapter_name=[model_name] * len(datas["question"]),
                                                **{"question": datas["question"]
                                                    , "requirement": datas["requirement"]
                                                    , "context": datas["context"]})
            datas.update({"pred": predictions})
            for i in range(len(predictions)):
                results.append({
                    "question": datas["question"][i],
                    "requirement": datas["requirement"][i],
                    "context": datas["context"][i],
                    "output": datas["output"][i],
                    "pred": datas["pred"][i]
                })

        # results 保存到output_file
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"Results saved to {output_file}")


def run_rag_prediction(data_dir,
                       output_dir,
                       template_file,
                       model_name="names",
                       max_samples=None,
                       model_invoke=get_qwen_response):

    def shuffle_context(context_str):
        context_list = context_str.split("【标题】：")

        shuffle(context_list)
        return "【标题】：" + "\n\n【标题】：".join(context_list)

    os.makedirs(output_dir, exist_ok=True)

    rag_agent = BaseLiteLLMAgent(template_file=template_file, model_invoke=model_invoke)

    for data_file in os.listdir(data_dir):
        output_file = data_file.replace(".json", f"_output.json")
        data_file = os.path.join(data_dir, data_file)
        model_output_dir = output_dir + f"/{model_name}/"
        os.makedirs(model_output_dir, exist_ok=True)
        output_file = os.path.join(model_output_dir, output_file)
        print(f"Processing data file: {data_file}")
        results = []
        for i, datas in tqdm(
                enumerate(batch_dataset_iterator(data_file, batch_size=4, max_samples=max_samples)),
                desc=f"processing {data_file.split('/')[-1]}"):
            # shuffled_contexts = [shuffle_context(x) for x in datas["context"]]
            predictions = rag_agent.para_invoke(adapter_name=[model_adapter_name_map[model_name]] * len(datas["question"]),
                                                **{"question": datas["question"]
                                                    , "requirement": datas["requirement"]
                                                    , "context": datas["context"]})
            datas.update({"pred": predictions})
            for j in range(len(predictions)):
                results.append({
                    "question": datas["question"][j],
                    "requirement": datas["requirement"][j],
                    "context": datas["context"][j],
                    "output": datas["output"][j],
                    "pred": datas["pred"][j]
                })
            if i % 200 == 1:
                # results 保存到output_file
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=4)
                print(f"question:{results[-1]['question']}")
                print(f"output:{results[-1]['output']}")
                print(f"pred:{results[-1]['pred']}")
                print(f"temp {i} Results saved to {output_file}")

        # results 保存到output_file
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"Results saved to {output_file}")


if __name__ == '__main__':
    # chatglm前后评估
    run_rag_prediction(
        data_dir="dataset/evaluation_dataset",
        output_dir="output/evaluation_dataset",
        template_file="template/template.json",
        model_name="chatglm",
        max_samples=None,
        model_invoke=get_chatglm_response
    )
    run_rag_prediction(
        data_dir="dataset/evaluation_dataset",
        output_dir="output/evaluation_dataset",
        template_file="template/template.json",
        model_name="chatglm-rag-0515",
        max_samples=None,
        model_invoke=get_chatglm_response
    )
    run_rag_prediction(
        data_dir="dataset/evaluation_dataset",
        output_dir="output/evaluation_dataset",
        template_file="template/template.json",
        model_name="chatglm-rag-0515-dpo",
        max_samples=None,
        model_invoke=get_chatglm_response
    )
    # run_rag_prediction(
    #     data_dir="dataset/evaluation_dataset",
    #     output_dir="output/evaluation_dataset",
    #     template_file="template/template.json",
    #     model_name="chatglm-rag-0515",
    #     max_samples=10,
    #     model_invoke=get_chatglm_response
    # )
    # 跑训练集的预测, 用于构建comparison数据
    # run_rag_prediction(
    #     data_dir="dataset/train_dataset",
    #     output_dir="output/train_dataset",
    #     template_file="template/template.json",
    #     model_name="chatglm-rag-0515",
    #     max_samples=None,
    #     model_invoke=get_chatglm_response
    # )
    # run_rag_prediction(
    #     data_dir="dataset/train_dataset",
    #     output_dir="output/train_dataset",
    #     template_file="template/template.json",
    #     model_name="qwen-rag-0515",
    #     max_samples=None,
    #     model_invoke=get_qwen_response
    # )

    # 跑测试集的预测，用于评估
    run_rag_evaluation(
        data_dir="dataset/",
        output_dir="output",
        template_file="template/template.json",
        model_name="original",
        max_samples=None,
        model_invoke=get_chatglm_response
    )
    run_rag_evaluation(
        data_dir="dataset/",
        output_dir="output",
        template_file="template/template.json",
        model_name="default",
        max_samples=None,
        model_invoke=get_qwen_response
    )
    run_rag_evaluation(
        data_dir="dataset/",
        output_dir="output",
        template_file="template/template.json",
        model_name="rag2",
        max_samples=None,
        model_invoke=get_qwen_response
    )
    run_rag_evaluation(
        data_dir="dataset/",
        output_dir="output",
        template_file="template/template.json",
        model_name="rag3",
        max_samples=None,
        model_invoke=get_qwen_response
    )
    run_rag_evaluation(
        data_dir="dataset/",
        output_dir="output",
        template_file="template/template.json",
        model_name="rag4",
        max_samples=None,
        model_invoke=get_qwen_response
    )
