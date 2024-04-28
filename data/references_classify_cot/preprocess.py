import json
import random
import re
from typing import List

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

# # 20230101
# template = """```
# {context}
# ```
#
# {question}"""
#
# system = """你是就职于中国平安人寿保险公司的保险专家，你的名字叫做安安。你需要使用由三个`符号包括的内容（即检索资料）回答问题或者执行指令。
# 如果检索资料内存在与问题几乎相同的标题，你需要复制检索资料内容的原文进行回答。
# 如果检索资料内存在与问题非常相似的标题，你需要模仿检索资料的内容的结构，使用相关内容的原文进行回答。
# 如果检索资料内不存在相同或相似的标题，但是内容里提供了足够回答问题所需要的知识，你需要有选择的选取相关的内容归纳总结，并回答问题。
# 如果检索资料无法提供足够回答问题所需要的知识，你需要拒绝回答，并明确指出你缺少哪些知识。
# 你必须保证你的回答里的所有内容都有检索资料的支持，因此你只能使用给定的检索资料进行回答，不允许进行额外的推理、猜测、延伸、想象。"""

# 20230201
generate_answer_prompt = "保险参考资料：\n```\n{context}\n```\n问题：{question}\n\n根据所提供的保险参考资料和问题，需要判断该保险参考资料属于以下五种类型中的哪一种：类型A、类型B、类型C、类型D、类型E。在回答问题之前，必须提供一个合理的、符合逻辑的推理过程。推理过程应该清晰地解释为什么参考资料属于某种类型。然后给出最终答案。\n类型A、B、C、D、E是用来描述参考资料与提出的问题之间关联性和有效性的不同级别。以下是每种类型的详细解释：        \n- **类型A：** 参考资料与问题完全相关，并且可以直接提供问题的答案。这种资料完全匹配问题的需求，提供了详尽的信息，直接解决了提出的问题。\n        \n- **类型B：** 参考资料内容与问题的核心相关，这种资料没有直接回答问题，而是提供了回答问题所需的核心知识。可以对问题的理解或解决提供较大的帮助。\n \n- **类型C：** 参考资料部分内容与问题相关，同时也包括了其他无关信息。这种资料不足以回答问题，但可能为问题提供部分有用信息, 对问题的理解或解决提供少量的帮助。\n\n- **类型D：** 参考资料在语义上可能与问题相似，即可能包含一些看似相关的词汇或概念，但实际上这些资料与问题的核心内容无关，因此不能提供问题的解答。它们可能误导或引起混淆。\n        \n - **类型E：** 参考资料与问题语义上不相似，且与问题本质上无关，因此无法帮助回答问题。这种类型的资料通常与问题完全不相关，即使是在广泛的语境中也找不到任何联系。\n\n输出格式如下：\n##Reason:{推理过程} ##Answer:{A or B or C or D or E}"

system = "you are a helpful insurance assistant."


def correct_json_str(json_str):
    "修正可能错误的json string"
    json_str = json_str.replace('\'', '"')
    json_str = json_str.replace('“', '"').replace("”", '"')
    json_str = json_str.replace('‘', '"').replace("’", '"')
    json_str = json_str.replace("，", ',').replace("：", ':')
    json_str = re.sub(r'\{\{+', '{', json_str)
    json_str = re.sub(r'\}\}+', '}', json_str)
    json_str = re.sub(r'\,\,+', ',', json_str)
    json_str = re.sub(r'\:\:+', ':', json_str)
    return json_str


def extract_and_correct_json(text):
    "提取答案中的json"
    match = re.search(r'\{.*?\}', text)
    if match:
        json_str = match.group()
        json_str = correct_json_str(json_str)
        try:
            json_data = json.loads(json_str)
            return json_data
        except json.JSONDecodeError:
            return {}
    else:
        return {}


def preprocess_jsonl_data_and_convert_to_dataset(file_name: [List[str], str], dev_ratio=0):
    train_datas = []
    dev_datas = []

    if isinstance(file_name, str):
        data_list = pd.read_json(file_name, lines=True).to_dict(orient="records")
    elif isinstance(file_name, list):
        data_list = []
        for file in file_name:
            data_list.extend(pd.read_json(file, lines=True).to_dict(orient="records"))
    else:
        raise ValueError("file_name must be a string or a list of strings")

    random.shuffle(data_list)
    train_nums = int(len(data_list) * (1 - dev_ratio))

    print(f"Load train: {train_nums} data from {file_name}")
    print(f"Load dev: {len(data_list) - train_nums} data from {file_name}")

    for i, data in tqdm(enumerate(data_list)):
        question = data["question"]
        context = data["context"]
        output = data["output"]
        type = data["type"]
        # prompt = generate_answer_prompt.format(**{"question": question, "context": context})
        prompt = generate_answer_prompt.replace("{context}", context).replace("{question}", question)
        if i <= train_nums:
            train_datas.append({
                "system": system,
                'instruction': prompt,
                'input': "",
                'output': output,
                'history': []
            })
        else:
            dev_datas.append({
                "system": system,
                'instruction': prompt,
                'input': "",
                'output': output,
                'history': []
            })

    return train_datas, dev_datas


if __name__ == '__main__':
    train_datas, dev_datas = preprocess_jsonl_data_and_convert_to_dataset(
        ["rag_dataset_0415_step1_fix.jsonl", "非核心_0415_step1_fix.jsonl"], 0.01)
    print("done")
    print(f"train datasets: {len(train_datas)}")
    print(f"eval datasets: {len(dev_datas)}")
    json.dump(train_datas, open("train_references_classify_cot_0415.json", "w"), ensure_ascii=False, indent=4)
    json.dump(dev_datas, open("eval_references_classify_cot_0415.json", "w"), ensure_ascii=False, indent=4)
    # json.dump(dev_datas, open("dev_askbob_0222_ft.json", "w"), ensure_ascii=False, indent=4)

    # train_datas, dev_datas = preprocess_askbob_data("rag_dataset_0201.json", replace_slash=False, shuffle_context=False)
    # print("done")
    # json.dump(train_datas, open("train_askbob_0201_ff.json", "w"), ensure_ascii=False, indent=4)
    # json.dump(dev_datas, open("dev_askbob_0201_ff.json", "w"), ensure_ascii=False, indent=4)
