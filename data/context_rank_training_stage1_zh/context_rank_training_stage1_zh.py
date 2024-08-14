import json
import random
from copy import deepcopy

import datasets
from typing import Any, Dict, List

from transformers.utils import logging

_DESCRIPTION = "RAG dataset with context ranking"
_CITATION = ""
_HOMEPAGE = ""
_LICENSE = ""

_URL = "./"
_URLS = {
    "train": [
        _URL + "search_ranking_train.json",
        _URL + "zhidao_ranking_train.json"
    ],
    "test": [],
}

logger = logging.get_logger(__name__)

class ContextRankDataset(datasets.GeneratorBasedBuilder):
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
            for example in example_dataset:
                yield key, example
                key += 1