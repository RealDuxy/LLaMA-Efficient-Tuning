# -*- encoding: utf-8 -*-
"""
@File    : main.py
@Time    : 18/5/2024 18:50
@Author  : Duxy
@Email   : du.xi.yang@qq.com
@Software: PyCharm
"""
import json

import pandas as pd

from data.nq_bm25_top100_kilt.nq_bm25_top100_kilt import NQRAGDataset, _URLS
from evaluation.src.run_evaluation import token_len, describe

# import json
#
# from data.instruction_only_rag.instruction_only_rag import InstructionOnlyDataset
# from data.qwen_rag_comparison.qwen_rag_comparison import QwenRAGComparisonDataset, _URL
# # dataset = QwenRAGComparisonDataset()
# # comparison_datas = [new_example
# #                     for key, new_example in dataset._generate_examples(filepaths=_URL)]
# #
# # json.dump(comparison_datas,
# #           open("qwen_rag_dpo.json", "w", encoding="utf-8"),
# #           ensure_ascii=False, indent=4)
#
# # from data.chatglm_rag_comparison.chatglm_rag_comparison import ChatGLMRAGComparisonDataset, _URL
# # dataset = ChatGLMRAGComparisonDataset()
#
# from data.qwen_rag_comparison.qwen_rag_comparison import QwenRAGComparisonDataset, _URL
# dataset = QwenRAGComparisonDataset()
#
# comparison_datas = [new_example
#                     for key, new_example in dataset._generate_examples(filepaths=_URL)]
#
# print(len(comparison_datas))
# json.dump(comparison_datas,
#           open("data/qwen_rag_comparison_0527_10p.json", "w", encoding="utf-8"),
#           ensure_ascii=False, indent=4)


# dataset = RAGDataset()
# for key, new_example in dataset._generate_examples(filepaths=_URLS["train"]):
#     if key % 10000 == 0:
#         print(key)
# print(key)


dataset = NQRAGDataset("train")
all_examples = []
all_data = []
for key, new_example in dataset._generate_examples(filepaths=_URLS["train"]):
    all_examples.append(new_example["system"]+new_example["instruction"]+new_example["output"])
    all_data.append(new_example)
json.dump(all_data, open("data/nq_bm25_top100_kilt/nq_bm25_top100_kilt_train.json", "w", encoding="utf-8"), indent=4)

token_length = token_len(all_examples)
print(describe(token_length))

# import pandas as pd
#
# splits = {'dev': 'data/dev-00000-of-00001-365806a8fce42050.parquet', 'test_without_answers': 'data/test_without_answers-00000-of-00001-49c3b81d44c12b52.parquet', 'train': 'data/train-00000-of-00001-9a5d4b2855a1daa0.parquet'}
# df = pd.read_parquet("hf://datasets/iohadrubin/nq_bm25_top100_kilt/" + splits["train"]).to_dict("records")


