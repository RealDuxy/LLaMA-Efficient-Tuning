# -*- encoding: utf-8 -*-
"""
@File    : main.py
@Time    : 18/5/2024 18:50
@Author  : Duxy
@Email   : du.xi.yang@qq.com
@Software: PyCharm
"""
import json

# from data.qwen_rag_comparison.qwen_rag_comparison import QwenRAGComparisonDataset, _URL
# dataset = QwenRAGComparisonDataset()
# comparison_datas = [new_example
#                     for key, new_example in dataset._generate_examples(filepaths=_URL)]
#
# json.dump(comparison_datas,
#           open("qwen_rag_dpo.json", "w", encoding="utf-8"),
#           ensure_ascii=False, indent=4)

# from data.chatglm_rag_comparison.chatglm_rag_comparison import ChatGLMRAGComparisonDataset, _URL
# dataset = ChatGLMRAGComparisonDataset()

from data.qwen_rag_comparison.qwen_rag_comparison import QwenRAGComparisonDataset, _URL
dataset = QwenRAGComparisonDataset()

comparison_datas = [new_example
                    for key, new_example in dataset._generate_examples(filepaths=_URL)]

print(len(comparison_datas))
json.dump(comparison_datas,
          open("data/qwen_rag_comparison_100p.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=4)