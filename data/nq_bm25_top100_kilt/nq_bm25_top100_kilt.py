import json
import os
import random
from copy import deepcopy

import datasets
from typing import List

import pandas as pd
from transformers.utils import logging

_HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://huggingface.co")
_DESCRIPTION = "general Chinese RAG dataset "
_CITATION = ""
_HOMEPAGE = "{}/datasets/iohadrubin/nq_bm25_top100_kilt".format(_HF_ENDPOINT)
_LICENSE = "mit"
_URL = "{}/datasets/iohadrubin/nq_bm25_top100_kilt/resolve/main/data/".format(_HF_ENDPOINT)

_URLS = {
    "train": [
        _URL + "train-00000-of-00001-9a5d4b2855a1daa0.parquet",
    ],
    "dev": [
        _URL + "dev-00000-of-00001-365806a8fce42050.parquet",
    ],
    "test": [
        _URL + "test_without_answers-00000-of-00001-49c3b81d44c12b52.parquet",
    ],
}

logger = logging.get_logger(__name__)
template = json.load(open("data/nq_bm25_top100_kilt/template.json", "r", encoding="utf-8"))
class NQRAGDataset(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("0.0.0")
    def _info(self) -> datasets.DatasetInfo:
        features = datasets.Features({
            "system": datasets.Value("string"),
            "instruction": datasets.Value("string"),
            "input": datasets.Value("string"),
            "output": datasets.Value("string")
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
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepaths": file_path["dev"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepaths": file_path["test"]})
        ]

    def _generate_examples(self, filepaths: List[str]):
        key = 0
        for filepath in filepaths:
            example_dataset = pd.read_parquet(filepath).to_dict("records")
            prompt_templates = deepcopy(template)
            system = prompt_templates["history"][0]["content"]
            prompt = prompt_templates["prompt"]
            for example in example_dataset:
                question = example["question"]
                answer =  json.loads(example["output"])[0]
                if len(answer) != 2:
                    continue
                answer_content = answer["answer"]
                answer_passage_title = answer["provenance"][0]["title"]
                passages_list = example["ctxs"].tolist()

                passages_str = ""
                index = 1
                for passages in passages_list:
                    if passages["contents"].startswith(answer_passage_title):
                        passages_str +=  f"Passage {index}: {passages['contents']}\n\n"
                        index += 1
                    if index > 10:
                        break

                new_example = {
                    "system": system.replace("{passage}", passages_str),
                    "instruction": prompt.replace("{question}", question),
                    "input": "",
                    "output": answer_content
                }
                yield key, new_example
                key += 1