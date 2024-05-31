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
import time
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
    "qwen-rag-0527-exp2": "default",
    "qwen-rag-0529-simpo-exp2": "rag1",
    "qwen-rag-0529-simpo-exp3": "rag2"
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
                       model_invoke=get_qwen_response,
                       sorted_by_output=False):

    def shuffle_context(context_str):
        context_list = context_str.split("【标题】：")

        shuffle(context_list)
        return "【标题】：" + "\n\n【标题】：".join(context_list)

    os.makedirs(output_dir, exist_ok=True)

    rag_agent = BaseLiteLLMAgent(template_file=template_file, model_invoke=model_invoke)


    for data_file in os.listdir(data_dir):
        time_start = time.time()
        # 过滤文件
        if "train_0524" not in data_file: continue

        output_file = data_file.replace(".json", f"_output.json")
        data_file = os.path.join(data_dir, data_file)
        model_output_dir = output_dir + f"/{model_name}/"
        os.makedirs(model_output_dir, exist_ok=True)
        output_file = os.path.join(model_output_dir, output_file)
        print(f"Processing data file: {data_file}")
        results = []
        for i, datas in tqdm(
                enumerate(batch_dataset_iterator(data_file, batch_size=4, max_samples=max_samples,sorted_by_output=sorted_by_output)),
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
        time_end = time.time()
        print(f"{model_name}-{data_file.split('/')[-1]} time cost: {(time_end - time_start)}")
        print(f"{model_name}-{data_file.split('/')[-1]} time cost per cost: {(time_end - time_start) / (4*(i+1))}")
        print(f"{model_name}-{data_file.split('/')[-1]} time cost per batch: {(time_end - time_start) / (i+1)}")

def run_rag_comparison(data_dir,
                       output_dir,
                       template_file,
                       model_names=["qwen"],
                       max_samples=None,
                       model_invoke=get_qwen_response,
                       sorted_by_output=False):

    os.makedirs(output_dir, exist_ok=True)

    rag_agent = BaseLiteLLMAgent(template_file=template_file, model_invoke=model_invoke)

    for data_file in os.listdir(data_dir):
        time_start = time.time()
        output_file = data_file.replace(".json", f"_output.json")
        data_file = os.path.join(data_dir, data_file)
        model_output_dir = output_dir + f"/0527_comparison/"
        os.makedirs(model_output_dir, exist_ok=True)
        output_file = os.path.join(model_output_dir, output_file)
        print(f"Processing data file: {data_file}")
        results = []
        for i, datas in tqdm(
                enumerate(batch_dataset_iterator(data_file, batch_size=4, max_samples=max_samples,sorted_by_output=sorted_by_output)),
                desc=f"processing {data_file.split('/')[-1]}"):
            # shuffled_contexts = [shuffle_context(x) for x in datas["context"]]
            for model_name in model_names:
                predictions = rag_agent.para_invoke(adapter_name=[model_adapter_name_map[model_name]] * len(datas["question"]),
                                                    **{"question": datas["question"]
                                                        , "requirement": datas["requirement"]
                                                        , "context": datas["context"]})

                datas.update({f"{model_name}_prediction": predictions})

            for j in range(len(predictions)):
                tmp_result = {
                    "question": datas["question"][j],
                    "requirement": datas["requirement"][j],
                    "context": datas["context"][j],
                    "output": datas["output"][j]
                }
                for model_name in model_names:
                    tmp_result.update({f"{model_name}_prediction": datas[f"{model_name}_prediction"][j]})
                results.append(tmp_result)

            if i % 5 == 1:
                # results 保存到output_file
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=4)
                print(f"question:{results[-1]['question']}")
                print(f"output:{results[-1]['output']}")
                for model_name in model_names:
                    print({f"{model_name}_prediction": results[-1][f"{model_name}_prediction"]})
                print(f"temp {i} Results saved to {output_file}")

        # results 保存到output_file
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"Results saved to {output_file}")
        time_end = time.time()
        print(f"{data_file.split('/')[-1]} time cost: {(time_end - time_start)}")
        print(f"{data_file.split('/')[-1]} time cost per cost: {(time_end - time_start) / (4*(i+1))}")
        print(f"{data_file.split('/')[-1]} time cost per batch: {(time_end - time_start) / (i+1)}")

if __name__ == '__main__':
    time_start = time.time()
    run_rag_comparison(
        data_dir="dataset/evaluation_dataset",
        output_dir="output/evaluation_dataset",
        template_file="template/template.json",
        model_names=["qwen","qwen-rag-0527-exp2","qwen-rag-0529-simpo-exp2","qwen-rag-0529-simpo-exp3"],
        max_samples=None,
        model_invoke=get_qwen_response
    )
    time_end = time.time()
    print(f"total time cost: {(time_end - time_start)}")
