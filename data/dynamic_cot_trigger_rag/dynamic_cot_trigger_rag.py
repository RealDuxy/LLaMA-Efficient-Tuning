import json
import random
from copy import deepcopy

import datasets
from typing import Any, Dict, List

_DESCRIPTION = "RAG dataset with dynamic CoT trigger"
_CITATION = ""
_HOMEPAGE = ""
_LICENSE = ""

_URL = "./"
_URLS = {
    "train": [
        _URL + "train_0619_dynamic_cot_trigger.json",
        # _URL + "train_0619_fix_cot_trigger.json",
        _URL + "train_0619_instruction_only.json"
    ],
    "test": [
        _URL + "eval_dynamic_cot_trigger.json",
        _URL + "eval_dynamic_cot_trigger_all_negative.json",
        _URL + "eval_fix_cot_trigger.json",
        _URL + "eval_fix_cot_trigger_all_negative.json",
        _URL + "eval_instruction_only.json",
        _URL + "eval_instruction_only_all_negative.json"
    ],
}

template = json.load(open("data/dynamic_cot_trigger_rag/template_0620.json", "r", encoding="utf-8"))

class DynamicCoTDataset(datasets.GeneratorBasedBuilder):
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
            with open(filepath, "r", encoding="utf-8") as f:
                for row in f:
                    example = json.loads(row)
                    system = template["history"][0]["content"]
                    prompt = template["prompt"]
                    question = example["question"]
                    if question[-1] not in ["？", "。", "！", "?", ".", "!"]:
                        question += "？"
                    requirement = example["requirement"].replace("\n", "")
                    output = example["output"]
                    context = example["contexts"]
                    random.shuffle(context)
                    context = "\n\n".join(context)
                    new_example = {
                        "system": system,
                        "instruction": prompt.replace("{question}", question).replace("{requirement}", requirement).replace("{context}", context),
                        "input": "",
                        "output": output,
                        "history": []
                    }
                    yield key, new_example
                    key += 1