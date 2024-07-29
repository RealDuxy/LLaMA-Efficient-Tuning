import json
import os
import random
from copy import deepcopy

import datasets
from typing import List

from transformers.utils import logging

_HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://huggingface.co")
_DESCRIPTION = "general Chinese stage2 insurance RAG dataset "
_CITATION = ""
_HOMEPAGE = ""
_LICENSE = "mit"
_URL = "./"

_URLS = {
    "train": [
        _URL + "train_0729_instruction_only.json",
        _URL + "train_0729_fix_cot_trigger.json",
        _URL + "train_0729_dynamic_cot_trigger.json"
    ],
    "test": []
}

logger = logging.get_logger(__name__)
template = json.load(open("data/anan_rag_training_stage2_zh/template_qa.json", "r", encoding="utf-8"))
class AnanStage2RAGDataset(datasets.GeneratorBasedBuilder):
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
                question = example["question"]
                if question[-1] not in ["？", "。", "！", "?", ".", "!"]:
                    question += "？"
                requirement = example["requirement"].replace("\n", "")
                output = example["output"]
                if example["is_positive"]:
                    context = example["oracles"] + example["distractors"]
                else:
                    context = example["distractors"]
                if isinstance(context, list):
                    random.shuffle(context)
                    context = "\n```\n```\n".join(context)
                new_example = {
                    "system": system.replace("{context}", context),
                    "instruction": prompt.replace("{question}", question).replace("{requirement}", requirement),
                    "input": "",
                    "output": output
                }
                yield key, new_example
                key += 1