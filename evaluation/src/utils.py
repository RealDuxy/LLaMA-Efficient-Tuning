# -*- encoding: utf-8 -*-
"""
@File    : utils.py
@Time    : 15/5/2024 11:00
@Author  : Duxy
@Email   : du.xi.yang@qq.com
@Software: PyCharm
"""
import json
import random
from copy import deepcopy
from typing import Dict, Any

import requests
from tenacity import wait_random_exponential, retry, stop_after_attempt


@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(10))
def get_qwen_response(history, prompt, **kwargs) -> str:
    """
    获取litemqwen推理服务的回复
    Args:
        self ():
        history ():
        prompt ():
        kwargs ():

    Returns:

    """
    url = "http://localhost:8081/chat"
    record_id = kwargs.get("record_id", random.randint(12347890, 99999999))
    headers = {"Content-Type": "application/json", "cache-control": "no-cache"}
    temperature = kwargs.get("temperature", 0.3)
    adapter_name = kwargs.get('adapter_name', "")
    top_k = kwargs.get("top_k", 50)
    skip_lora = (adapter_name == "original" or adapter_name == "")
    seed = kwargs.get("seed", 42)
    prefix_token_ids = kwargs.get("prefix_token_ids", [])

    input_data = {
        "query": prompt,
        "history": history,
        "request_id": record_id,
        "gen_kwargs": {"seed": seed,
                       "prefix_token_ids": prefix_token_ids,
                       "temperature": temperature,
                       "skip_lora": skip_lora,
                       "adapter_name": adapter_name,
                       "top_k": top_k,
                       "return_raw": True}}
    response = requests.post(url=url, headers=headers, data=json.dumps(input_data))
    if response.status_code != 200:
        return "查询结果出错"
    else:
        resp = response.json()
        return resp['response']

@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(10))
def get_chatglm_response(history, prompt, **kwargs) -> str:
    """
    获取litemqwen推理服务的回复
    Args:
        self ():
        history ():
        prompt ():
        kwargs ():

    Returns:

    """
    url = "http://localhost:8081/chat"
    record_id = kwargs.get("record_id", random.randint(12347890, 99999999))
    headers = {"Content-Type": "application/json", "cache-control": "no-cache"}
    temperature = kwargs.get("temperature", 0.3)
    adapter_name = kwargs.get('adapter_name', "")
    top_k = kwargs.get("top_k", 50)
    skip_lora = (adapter_name == "original" or adapter_name == "")
    seed = kwargs.get("seed", 42)
    prefix_token_ids = kwargs.get("prefix_token_ids", [])

    input_data = {
        "query": prompt,
        "history": history,
        "request_id": record_id,
        "gen_kwargs": {"seed": seed,
                       "prefix_token_ids": prefix_token_ids,
                       "temperature": temperature,
                       "skip_lora": skip_lora,
                       "adapter_name": adapter_name,
                       "top_k": top_k,
                       "return_raw": True}}
    response = requests.post(url=url, headers=headers, data=json.dumps(input_data))
    if response.status_code != 200:
        return "查询结果出错"
    else:
        resp = response.json()
        return resp['response']


def batch_dataset_iterator(filepath, batch_size=4, max_samples=None) -> Dict[str, Any]:
    example_dataset = json.load(open(filepath, "r", encoding="utf-8"))
    if max_samples:
        example_dataset = example_dataset[:max_samples]

    batched_example_dataset = [example_dataset[i:i + batch_size] for i in range(0, len(example_dataset), batch_size)]
    for key, batched_samples in enumerate(batched_example_dataset):
        questions = [x["question"] for x in batched_samples]
        for i, question in enumerate(questions):
            if question[-1] not in ["？", "。", "！", "?", ".", "!"]:
                questions[i] = question + "？"
        requirements = [x["requirement"].replace("\n", "") for x in batched_samples]
        outputs = [x["output"] for x in batched_samples]
        contexts = [x["contexts"] for x in batched_samples]
        is_positives = [x["is_positive"] for x in batched_samples]
        new_example = {
            "question": questions,
            "requirement": requirements,
            "context": contexts,
            "output": outputs
        }
        yield new_example


def dataset_iterator(filepath: str, max_samples=None) -> Dict[str, Any]:
    example_dataset = json.load(open(filepath, "r", encoding="utf-8"))
    # 打印当前目录
    template = json.load(open("template/generate_answer.json", "r", encoding="utf-8"))
    prompt_templates = deepcopy(template)
    system = prompt_templates["history"][0]["content"]
    prompt = prompt_templates["prompt"]
    if max_samples:
        example_dataset = example_dataset[:max_samples]

    for key, sample in enumerate(example_dataset):
        question = sample["question"]
        if question[-1] not in ["？", "。", "！", "?", ".", "!"]:
            question += "？"
        requirement = sample["requirement"].replace("\n", "")
        output = sample["output"]
        context = sample["context"]
        is_positive = sample["is_positive"]
        new_example = {
            "system": system,
            "instruction": prompt.replace("{question}", question).replace("{requirement}", requirement).replace(
                "{context}", context),
            "input": "",
            "output": output,
            "history": []
        }
        yield new_example
