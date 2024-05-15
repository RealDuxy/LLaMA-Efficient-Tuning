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

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from base import BaseLiteLLMAgent
from utils import get_qwen_response, batch_dataset_iterator


def run_rag_evaluation(data_dir, output_dir,
                       template_file,
                       model_name="names",
                       max_samples=4,
                       model_invoke=get_qwen_response):

    rag_agent = BaseLiteLLMAgent(template_file=template_file, model_invoke=model_invoke)


    for data_file in os.listdir(data_dir):
        output_file = data_file.replace(".json", f"_output.json")
        data_file = os.path.join(data_dir, data_file)
        model_output_dir = output_dir + f"/{model_name}/"
        os.makedirs(model_output_dir, exist_ok=True)
        output_file = os.path.join(model_output_dir, output_file)

        print(f"data file: {data_file}")
        print(f"output file: {output_file}")
        results = []
        for datas in tqdm(batch_dataset_iterator(data_file, batch_size=4, max_samples=max_samples)):
            predictions = rag_agent.para_invoke(adapter_name=[model_name]*len(datas["question"]),
                                                **{"question":datas["question"]
                                                    ,"requirement":datas["requirement"]
                                                    ,"context": datas["context"]})
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



if __name__ == '__main__':
    run_rag_evaluation(
        data_dir="dataset/",
        output_dir="output",
        template_file="template/template.json",
        model_name="original",
        max_samples=8,
        model_invoke=get_qwen_response
    )
    run_rag_evaluation(
        data_dir="dataset/",
        output_dir="output",
        template_file="template/template.json",
        model_name="default",
        max_samples=8,
        model_invoke=get_qwen_response
    )
    run_rag_evaluation(
        data_dir="dataset/",
        output_dir="output",
        template_file="template/template.json",
        model_name="rag2",
        max_samples=8,
        model_invoke=get_qwen_response
    )
    run_rag_evaluation(
        data_dir="dataset/",
        output_dir="output",
        template_file="template/template.json",
        model_name="rag3",
        max_samples=8,
        model_invoke=get_qwen_response
    )
    run_rag_evaluation(
        data_dir="dataset/",
        output_dir="output",
        template_file="template/template.json",
        model_name="rag4",
        max_samples=8,
        model_invoke=get_qwen_response
    )
