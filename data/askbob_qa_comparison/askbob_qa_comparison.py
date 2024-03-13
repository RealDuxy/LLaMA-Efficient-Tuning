import json
import datasets
from typing import Any, Dict, List


_DESCRIPTION = "An example of dataset."
_CITATION = ""
_HOMEPAGE = ""
_LICENSE = ""
_URL = "./"

_URLS = {
    "train": _URL + "askbob_qa_comparison.json"
}

class AskBobQADataset(datasets.GeneratorBasedBuilder):

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
        # 使用dl_manager.download_and_extract来下载和解压（如果需要）文件
        downloaded_files = dl_manager.download_and_extract(_URLS)

        # 返回SplitGenerator实例的列表，分别为训练集和验证集指定正确的文件路径
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": downloaded_files["train"]  # 为训练集指定文件路径
                }
            ),
            # datasets.SplitGenerator(
            #     name=datasets.Split.VALIDATION,
            #     gen_kwargs={
            #         "filepath": downloaded_files["dev"]  # 为验证集指定文件路径，注意这里使用的键与_URLS中的键匹配
            #     }
            # )
        ]

    def _generate_examples(self, filepath: str) -> Dict[int, Dict[str, Any]]:
        example_dataset = json.load(open(filepath, "r", encoding="utf-8"))
        for key, example in enumerate(example_dataset):
            yield key, example
