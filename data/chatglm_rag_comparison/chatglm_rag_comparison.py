import json
from copy import deepcopy

import datasets
from typing import Any, Dict, List

_DESCRIPTION = "RAG dataset with dynamic CoT trigger"
_CITATION = ""
_HOMEPAGE = ""
_LICENSE = ""
_URL_DIR = "data/chatglm_rag_comparison/"
_URL = [
    _URL_DIR+"train_instruction_only_comparison.json",
    _URL_DIR+"train_fix_cot_trigger_comparison.json",
    _URL_DIR+"train_dynamic_cot_trigger_comparison.json"
]

template = json.load(open("data/dynamic_cot_trigger_rag/template.json", "r", encoding="utf-8"))

class ChatGLMRAGComparisonDataset(datasets.GeneratorBasedBuilder):
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

    # def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
    #     file_path = dl_manager.download(_URL)
    #     return [datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": file_path})]

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        file_paths = dl_manager.download_and_extract(_URL)  # 确保这是一个文件列表
        return [datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepaths": file_paths})]

    def _generate_examples(self, filepaths: List[str]) -> Dict[int, Dict[str, Any]]:
        prompt_templates = deepcopy(template)
        system = prompt_templates["history"][0]["content"]
        prompt = prompt_templates["prompt"]

        key = 0
        for filepath in filepaths:
            example_dataset = json.load(open(filepath, "r", encoding="utf-8"))
            for example in example_dataset:
                question = example["question"]
                if question[-1] not in ["？", "。", "！", "?", ".", "!"]:
                    question += "？"
                requirement = example["requirement"].replace("\n", "")
                output = example["output"]
                context = example["contexts"]
                new_example = {
                    "system": system,
                    "instruction": prompt.replace("{question}", question).replace("{requirement}", requirement).replace("{context}", context),
                    "input": "",
                    "output": output,
                    "history": []
                }
                yield key, new_example
                key += 1


