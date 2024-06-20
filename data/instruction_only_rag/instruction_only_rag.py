import json
import random
from copy import deepcopy

import datasets
from typing import Any, Dict, List

_DESCRIPTION = "RAG dataset"
_CITATION = ""
_HOMEPAGE = ""
_LICENSE = ""
# _URL = "train_0524_instruction_only.json"
_URL = "train_0619_instruction_only.json"

template = json.load(open("data/instruction_only_rag/template.json", "r", encoding="utf-8"))

class InstructionOnlyDataset(datasets.GeneratorBasedBuilder):
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

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        file_path = dl_manager.download(_URL)
        return [datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": file_path})]

    def _generate_examples(self, filepath: str) -> Dict[int, Dict[str, Any]]:
        example_dataset = json.load(open(filepath, "r", encoding="utf-8"))
        # prompt_templates = deepcopy(template)
        # system = prompt_templates["history"][0]["content"]
        # prompt = prompt_templates["prompt"]
        system = "你是安安，是由中国平安人寿保险有限公司开发和提供的智能保险专家。你的任务是根据检索到的资料回答用户的问题。"
        prompt = "**检索到的资料**：\n```\n{context}\n```\n**问题**：{question}{requirement}"
        for key, example in enumerate(example_dataset):
            question = example["question"]
            if question[-1] not in ["？", "。", "！", "?", ".", "!"]:
                question += "？"
            requirement = example["requirement"].replace("\n", "")
            output = example["output"]
            context = example["contexts"]
            random.shuffle(context)
            context = "\n\n".join(context)
            is_positive = example["is_positive"]
            new_example = {
                "system": system,
                "instruction": prompt.replace("{question}", question).replace("{requirement}", requirement).replace("{context}", context),
                "input": "",
                "output": output,
                "history": []
            }
            yield key, new_example
