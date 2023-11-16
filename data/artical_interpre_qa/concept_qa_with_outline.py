# -*- encoding: utf-8 -*-
"""
@File    : concept_qa_with_outline.py
@Time    : 9/11/2023 21:45
@Author  : Duxy
@Email   : du.xi.yang@qq.com
@Software: PyCharm
"""
import json
import datasets
from typing import List


_DESCRIPTION = "concept_qa_with_outline: 条款问答：根据检索信息和问题生成符合top1 outline的答案"

_CITATION = """RealDuxy"""


_BASE_DATA_URL = "https://huggingface.co/datasets/stingning/ultrachat/resolve/main/train_{idx}.jsonl"

_HOMEPAGE = ""
_LICENSE = ""
_URL = "0924/output_threshold_25/scn123_理念导入_train.json"

class ConceptQAOutline(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("0.0.0")

    def _info(self):
        features = datasets.Features({
            "conversations": [{"from": datasets.Value("string"), "value": datasets.Value("string")}]
        })
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        file_paths = dl_manager.download(_URL)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepaths": file_paths
                }
            )
        ]

    def _generate_examples(self, filepaths: List[str]):
        for filepath in filepaths:
            with open(filepath, "r", encoding="utf-8") as f:
                for row in f:
                    try:
                        data = json.loads(row)
                    except:
                        continue
                    key: int = data["id"]
                    content: List[str] = data["data"]
                    if len(content) % 2 == 1:
                        content.pop(-1)
                    if len(content) < 2:
                        continue
                    conversations = [{
                        "from": "human" if i % 2 == 0 else "gpt",
                        "value": content[i]
                    } for i in range(len(content))]
                    yield key, {
                        "conversations": conversations
                    }
