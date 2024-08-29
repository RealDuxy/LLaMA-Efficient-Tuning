# -*- encoding: utf-8 -*-
"""
@File    : test_inference.py
@Time    : 29/8/2024 16:43
@Author  : Duxy
@Email   : du.xi.yang@qq.com
@Software: PyCharm
"""
from openai import OpenAI

from openai import OpenAI
client = OpenAI(base_url="http://0.0.0.0:8000, api_key="")

completion = client.chat.completions.create(
  model="Qwen1.5-14B-Chat-GPTQ-Int8",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Who won the world series in 2020?"},
    {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
    {"role": "user", "content": "Where was it played?"}
  ]
)




from data.insurance_product_qa.insurance_product_qa import AnanStage2ProductQADataset

datasets = [
                AnanStage2ProductQADataset("test"),
                # AnanStage2RAGDataset("test"),
                # AnanStage2ContextRankDataset("test")
            ]


for dataset in datasets:
    from data.insurance_product_qa.insurance_product_qa import _URLS
    for key, new_example in dataset._generate_examples(filepaths=_URLS["test"]):
        all_examples.append(new_example["system"]+new_example["instruction"]+new_example["output"])
        all_data.append(new_example)



print(response.headers.get('x-request-id'))

# get the object that `chat.completions.create()` would have returned
completion = response.parse()
print(completion)
