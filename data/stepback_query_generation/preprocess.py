import json
import random
import re

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
generate_answer_prompt = """用户提问：{query}"""

system = """你是一个具有丰富保险专业知识和常识的专家，你的任务是根据用户提出的问题，退一步思考，并将用户提问改写成多个更加通用的、更简单的“回退问题”, 以Python List格式输出"""

tokenizer = AutoTokenizer.from_pretrained(
    "THUDM/chatglm3-6b", trust_remote_code=True
)

slash = "\n"


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

def preprocess_askbob_data(file_name, replace_slash=False, shuffle_context=True, dev_ratio=0.1):
    train_datas = []
    dev_datas = []
    with open(file_name, "r", encoding="utf-8") as f:
        data_list = json.load(f)
        random.shuffle(data_list)

        train_nums = int(len(data_list) * (1-dev_ratio))

        print(f"Load train: {train_nums} data from {file_name}")
        print(f"Load dev: {len(data_list) - train_nums} data from {file_name}")

        for i, data in tqdm(enumerate(data_list)):
            query_for_training = data["input"]
            output = data["step_back_prompt_question"]
            if not output:
                continue
            prompt = generate_answer_prompt.replace("{query}", query_for_training.strip())
            output = str(output)
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

def preprocess_stpbk_json_data(file_name, dev_ratio=0):
    train_datas = []
    dev_datas = []

    data_list = pd.read_excel(file_name).to_dict(orient="records")
    random.shuffle(data_list)

    train_nums = int(len(data_list) * (1-dev_ratio))

    print(f"Load train: {train_nums} data from {file_name}")
    print(f"Load dev: {len(data_list) - train_nums} data from {file_name}")

    for i, data in tqdm(enumerate(data_list)):
        query_for_training = data["input"]
        output = data["output"].strip()
        if not output:
            continue
        prompt = generate_answer_prompt.replace("{query}", query_for_training.strip())
        output = str(output)
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
    # train_data = preprocess_jsonline("1115_highQuality.json")
    # json.dump(train_data, open("train.json", "w"), ensure_ascii=False, indent=4)
    #
    # eval_data = preprocess_jsonline("1115_highQuality_eval.json")
    # json.dump(eval_data, open("dev.json", "w"), ensure_ascii=False, indent=4)

    # askbob_datas, high_quality_datas, normal_quality_datas = preprocess_llm_data("llm_data.json")
    # json.dump(askbob_datas, open("train_askbob.json", "w"), ensure_ascii=False, indent=4)
    # json.dump(high_quality_datas, open("train_high_quality.json", "w"), ensure_ascii=False, indent=4)
    # json.dump(normal_quality_datas, open("train_normal.json", "w"), ensure_ascii=False, indent=4)

    # train_datas, dev_datas = preprocess_askbob_data("rag_dataset_0201.json", replace_slash=True, shuffle_context=True)
    # print("done")
    # json.dump(train_datas, open("train_askbob_0201_tt.json", "w"), ensure_ascii=False, indent=4)
    # json.dump(dev_datas, open("dev_askbob_0201_tt.json", "w"), ensure_ascii=False, indent=4)

    # train_datas, dev_datas = preprocess_askbob_data("rag_dataset_0201.json", replace_slash=True, shuffle_context=False)
    # print("done")
    # json.dump(train_datas, open("train_askbob_0201_tf.json", "w"), ensure_ascii=False, indent=4)
    # json.dump(dev_datas, open("dev_askbob_0201_tf.json", "w"), ensure_ascii=False, indent=4)

    # train_datas, dev_datas = preprocess_askbob_data("rag_dataset_0201_gen_eva_1138.json", replace_slash=False, shuffle_context=True, dev_ratio=0)
    train_datas, dev_datas = preprocess_stpbk_json_data("0307_1000_gen.xlsx", 0)
    print("done")
    print(f"train datasets: {len(train_datas)}")
    print(f"eval datasets: {len(dev_datas)}")
    json.dump(train_datas, open("stepback_askbob_gpt4_1k_0307.json", "w"), ensure_ascii=False, indent=4)
    # json.dump(dev_datas, open("dev_askbob_0222_ft.json", "w"), ensure_ascii=False, indent=4)

    # train_datas, dev_datas = preprocess_askbob_data("rag_dataset_0201.json", replace_slash=False, shuffle_context=False)
    # print("done")
    # json.dump(train_datas, open("train_askbob_0201_ff.json", "w"), ensure_ascii=False, indent=4)
    # json.dump(dev_datas, open("dev_askbob_0201_ff.json", "w"), ensure_ascii=False, indent=4)
