# -*- encoding: utf-8 -*-
"""
@File    : rag.py
@Time    : 27/2/2024 16:25
@Author  : Duxy
@Email   : du.xi.yang@qq.com
@Software: PyCharm
"""
import pandas as pd
from tqdm import tqdm

from evaluation.stepback_query.base import BaseAgent
from evaluation.stepback_query.local_model_infer import get_chatglm_response


def run_test1():
    data_path = "评估集0304.xlsx"
    stepback_agent = BaseAgent(template_file="stepback.json", model_invoke=get_chatglm_response)

    df = pd.read_excel(data_path).to_dict("records")
    for i, line in tqdm(enumerate(df)):
        question = line["输入"]
        if isinstance(question, str):
            output = stepback_agent.invoke(query=question)
            df[i]["输出"] = output.strip()

if __name__ == '__main__':
    run_test1()