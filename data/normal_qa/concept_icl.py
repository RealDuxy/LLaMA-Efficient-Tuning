import json
from json import JSONDecodeError
from random import shuffle

import datasets
from typing import Any, Dict, List

from transformers import AutoTokenizer

from glmtuner.hparams import model_args, data_args

_DESCRIPTION = "An example of dataset for ChatGLM."
_CITATION = ""
_HOMEPAGE = ""
_LICENSE = ""
_URL = "0924/output_threshold_25/scn123_理念导入_train.json"

template = """‘’‘
{context}
’‘’

{question}"""
# template = """
# 你是平安寿险专业的保险销售专家，你能够非常有礼貌的帮助代理人或者客户解答各种跟保险相关的问题
# 参考以下信息：
# {context}
#
# 指令是：{question}
#
# 根据前面的参考信息列表，选取合适的内容，修复指令与参考信息列表内可能存在的冲突，根据参考信息的丰富程度自行选择是否生成内容，输出内容结构分条叙述，最后总结。要求输出内容跟指令相关。
# """

tokenizer = AutoTokenizer.from_pretrained(
    "THUDM/chatglm2-6b", trust_remote_code=True
)

class ExampleDataset(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("0.0.0")

    def _info(self) -> datasets.DatasetInfo:
        features = datasets.Features({
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
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": file_path
                }
            )
        ]


    def _generate_examples(self, filepath: str) -> Dict[int, Dict[str, Any]]:
        # f1 = open("/root/ChatGLM-Efficient-Tuning/data/concept_icl/scn1_理念导入_gpt4_len1500_all_train_0906.json", "r")
        # f2 = open("/root/ChatGLM-Efficient-Tuning/data/concept_icl/scn2_理念导入_gpt4_len1500_all_train_0906.json", "r")
        # lines1, line2 = f1.readlines(), f2.readlines()
        # for key, data in enumerate(lines1+line2):
        #     try:
        #         data = json.loads(data)
        #         question = data["input"]
        #         context = data["context"]
        #         prompt = template.replace("{question}", question).replace("{context}", context)
        #         yield key, {"instruction": prompt, "input": "", "output": data["output"], "history": []}
        #     except JSONDecodeError as e:
        #         print(f"第{key}行解析错误")
        #         pass
        #
        # f1.close()
        # f2.close()

        with open(filepath, "r") as f:
            for key, data in enumerate(f.readlines()):
                data = json.loads(data)

                history = [
                    ["你是就职于平安人寿保险公司的保险专家，你的名字叫安安。接下来我会为你提供参考资料，你需要阅读参考资料，然后礼貌且专业地回答我的问题或者执行我的指令。",
                     "好的，我是平安人寿保险专家安安，我会阅读您提供的参考资料，然后为您提供相关问题的答案或者执行您的指令。请您提供参考资料以及相关的问题或者指令。"]
                ]
                context = data["context"].replace("参考文档 [1]：", "").replace("参考文档 [2]：", "")
                input = data["input"]
                output = data["output"]

                context_ids = tokenizer.encode(text=context, add_special_tokens=False)
                templates_ids = tokenizer.encode(text=template, add_special_tokens=False)

                max_source_length = 1200
                max_context_length = max_source_length - 2 - len(templates_ids)
                if len(context_ids) > max_source_length - 2:  # gmask and sop tokens
                    context_ids = context_ids[:max_context_length]

                truncated_context = tokenizer.decode(context_ids, skip_special_tokens=True)

                prompt = template.replace("{question}", input).replace("{context}", truncated_context)

                yield key, {"instruction": prompt, "input": "", "output": data["output"], "history": history}
