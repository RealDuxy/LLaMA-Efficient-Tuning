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
import re
from copy import deepcopy
from typing import Dict, Any

import requests
import six
from rouge_chinese import rouge_score
from tenacity import wait_random_exponential, retry, stop_after_attempt
from tqdm import tqdm


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
    url = "http://127.0.0.1:8081/chat"
    record_id = kwargs.get("record_id", random.randint(1, 99999999))
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
                       # "max_new_tokens": 400,
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


def batch_dataset_iterator(filepath, batch_size=4, max_samples=None, sorted_by_output=False) -> Dict[str, Any]:
    example_dataset = json.load(open(filepath, "r", encoding="utf-8"))
    if max_samples:
        example_dataset = example_dataset[:max_samples]

    # sort by length to accelerate the batch inference
    # inference time cost mainly depends on output length
    if sorted_by_output:
        example_dataset = sorted(example_dataset, key=lambda x: len(x["output"]))
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


class Rouge:
    DEFAULT_METRICS = ["rouge-1", "rouge-2", "rouge-l"]
    AVAILABLE_METRICS = {
        "rouge-1": lambda hyp, ref, **k: rouge_score.rouge_n(hyp, ref, 1, **k),
        "rouge-2": lambda hyp, ref, **k: rouge_score.rouge_n(hyp, ref, 2, **k),
        "rouge-3": lambda hyp, ref, **k: rouge_score.rouge_n(hyp, ref, 3, **k),
        "rouge-4": lambda hyp, ref, **k: rouge_score.rouge_n(hyp, ref, 4, **k),
        "rouge-5": lambda hyp, ref, **k: rouge_score.rouge_n(hyp, ref, 5, **k),
        "rouge-l": lambda hyp, ref, **k:
        rouge_score.rouge_l_summary_level(hyp, ref, **k),
    }
    DEFAULT_STATS = ["r", "p", "f"]
    AVAILABLE_STATS = ["r", "p", "f"]

    def __init__(self, metrics=None, stats=None, return_lengths=False,
                 raw_results=False, exclusive=True):
        self.return_lengths = return_lengths
        self.raw_results = raw_results
        self.exclusive = exclusive

        if metrics is not None:
            self.metrics = [m.lower() for m in metrics]

            for m in self.metrics:
                if m not in Rouge.AVAILABLE_METRICS:
                    raise ValueError("Unknown metric '%s'" % m)
        else:
            self.metrics = Rouge.DEFAULT_METRICS

        if self.raw_results:
            self.stats = ["hyp", "ref", "overlap"]
        else:
            if stats is not None:
                self.stats = [s.lower() for s in stats]

                for s in self.stats:
                    if s not in Rouge.AVAILABLE_STATS:
                        raise ValueError("Unknown stat '%s'" % s)
            else:
                self.stats = Rouge.DEFAULT_STATS

    def cut_sent(self, para):
        para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)
        para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)
        para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)
        para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
        para = para.rstrip()
        return para.split("\n")

    def get_scores(self, hyps, refs, avg=False, ignore_empty=False):
        if isinstance(hyps, six.string_types):
            hyps, refs = [hyps], [refs]

        if ignore_empty:
            # Filter out hyps of 0 length
            hyps_and_refs = zip(hyps, refs)
            hyps_and_refs = [_ for _ in hyps_and_refs
                             if len(_[0]) > 0
                             and len(_[1]) > 0]
            hyps, refs = zip(*hyps_and_refs)

        assert (isinstance(hyps, type(refs)))
        assert (len(hyps) == len(refs))

        if not avg:
            return self._get_scores(hyps, refs)
        return self._get_avg_scores(hyps, refs)

    def _get_scores(self, hyps, refs):
        scores = []
        for hyp, ref in tqdm(zip(hyps, refs)):
            sen_score = {}

            hyp = [" ".join(_.split()) for _ in self.cut_sent(hyp) if len(_) > 0]
            ref = [" ".join(_.split()) for _ in self.cut_sent(ref) if len(_) > 0]

            for m in self.metrics:
                fn = Rouge.AVAILABLE_METRICS[m]
                sc = fn(
                    hyp,
                    ref,
                    raw_results=self.raw_results,
                    exclusive=self.exclusive)
                sen_score[m] = {s: sc[s] for s in self.stats}

            if self.return_lengths:
                lengths = {
                    "hyp": len(" ".join(hyp).split()),
                    "ref": len(" ".join(ref).split())
                }
                sen_score["lengths"] = lengths
            scores.append(sen_score)
        return scores

    def _get_avg_scores(self, hyps, refs):
        scores = {m: {s: 0 for s in self.stats} for m in self.metrics}
        if self.return_lengths:
            scores["lengths"] = {"hyp": 0, "ref": 0}

        count = 0
        for (hyp, ref) in zip(hyps, refs):
            hyp = [" ".join(_.split()) for _ in self.cut_sent(hyp) if len(_) > 0]
            ref = [" ".join(_.split()) for _ in self.cut_sent(ref) if len(_) > 0]

            for m in self.metrics:
                fn = Rouge.AVAILABLE_METRICS[m]
                sc = fn(hyp, ref, exclusive=self.exclusive)
                scores[m] = {s: scores[m][s] + sc[s] for s in self.stats}

            if self.return_lengths:
                scores["lengths"]["hyp"] += len(" ".join(hyp).split())
                scores["lengths"]["ref"] += len(" ".join(ref).split())

            count += 1
        avg_scores = {
            m: {s: scores[m][s] / count for s in self.stats}
            for m in self.metrics
        }

        if self.return_lengths:
            avg_scores["lengths"] = {
                k: scores["lengths"][k] / count
                for k in ["hyp", "ref"]
            }

        return avg_scores