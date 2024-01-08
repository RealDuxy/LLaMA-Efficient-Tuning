import json
import datasets
from typing import Any, Dict, List


_DESCRIPTION = "An example of dataset."
_CITATION = ""
_HOMEPAGE = ""
_LICENSE = ""
_URL = "./"

_URLS = {
    "train": _URL + "train_askbob.json",
    "test": _URL + "dev.json",
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

    # def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
    #     file_path = dl_manager.download(_URL)
    #     return [
    #         datasets.SplitGenerator(
    #             name=datasets.Split.TRAIN,
    #             gen_kwargs={
    #                 "filepath": file_path
    #             }
    #         )
    #     ]
    def _split_generators(self, dl_manager: datasets.DownloadManager):
        file_path = dl_manager.download(_URLS)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": file_path["train"]
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": file_path["test"]
                }
            )
        ]

    def _generate_examples(self, filepath: str) -> Dict[int, Dict[str, Any]]:
        example_dataset = json.load(open(filepath, "r", encoding="utf-8"))
        for key, example in enumerate(example_dataset):
            yield key, example
