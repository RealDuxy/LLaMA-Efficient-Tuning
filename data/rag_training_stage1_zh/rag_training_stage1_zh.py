import json
import os
import random
from copy import deepcopy

import datasets
from typing import List

from transformers.utils import logging

_HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://huggingface.co")
_DESCRIPTION = "general Chinese RAG dataset "
_CITATION = ""
_HOMEPAGE = "{}/datasets/Duxy/Chinese_RAG_SFT_training_data".format(_HF_ENDPOINT)
_LICENSE = "mit"
_URL = "{}/datasets/Duxy/Chinese_RAG_SFT_training_data/resolve/main/".format(_HF_ENDPOINT)

_URLS = {
    "train": [
        _URL + "squad-zen/train.json",
        _URL + "dureader_robust/train.json",
        _URL + "dureader2.0/search_train.json",
        _URL + "dureader2.0/zhidao_train.json",
        _URL + "SyntheConvQA/train.json",
        _URL + "afac2024/train.json",
        _URL + "WebCPM/train.json"
    ],
    "test": []
}

logger = logging.get_logger(__name__)
template = json.load(open("data/rag_training_stage1_zh/template_qa_short.json", "r", encoding="utf-8"))
class RAGDataset(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("0.0.0")
    def _info(self) -> datasets.DatasetInfo:
        features = datasets.Features({
            "system": datasets.Value("string"),
            "instruction": datasets.Value("string"),
            "input": datasets.Value("string"),
            "output": datasets.Value("string"),
            "history": datasets.Sequence(datasets.Sequence(datasets.Value("string")))
        })
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        file_path = dl_manager.download_and_extract(_URLS)
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepaths": file_path["train"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepaths": file_path["test"]}),
        ]

    def _generate_examples(self, filepaths: List[str]):
        key = 0
        for filepath in filepaths:
            example_dataset = json.load(open(filepath, "r", encoding="utf-8"))
            prompt_templates = deepcopy(template)
            system = prompt_templates["history"][0]["content"]
            prompt = prompt_templates["prompt"]
            for example in example_dataset:
                messages = example["messages"]
                question = messages[-1]["content"]

                if len( messages[:-1]) % 2 != 0:
                    print(f"file_path: {filepath}, row {key}")

                history = []
                history_messages = messages[:-1]
                for i in range(0, len(history_messages), 2):
                    history.append([history_messages[i]['content'], history_messages[i+1]['content']])

                context = example["oracles"] + example["distractors"]
                output = example["answers"][0]
                if isinstance(context, list):
                    random.shuffle(context)
                    context = "\n```\n```\n".join(context)
                new_example = {
                    "system": system.replace("{context}", context),
                    "instruction": prompt.replace("{question}", question),
                    "input": "",
                    "output": output,
                    "history": history
                }
                yield key, new_example
                key += 1