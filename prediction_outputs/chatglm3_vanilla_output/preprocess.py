import json
import random

from tqdm import tqdm
from transformers import AutoTokenizer

template = """```
{context}
```

{question}"""

system = """你是一名保险专家，你的名字是“安安”，你就职于中国平安人寿保险公司。你的任务是帮助代理人用户解决与保险相关的问题。"""

tokenizer = AutoTokenizer.from_pretrained(
    "THUDM/chatglm3-6b", trust_remote_code=True
)

def preprocess_jsonline(original_file, chatglm_file, is_train=True):
    cnt_drop = 0
    cnt_overall = 0
    cnt_zero = 0
    with open(chatglm_file, "r", encoding="utf-8") as f:
        chatglm_answer_list = {}
        for i, line in enumerate(f.readlines()):
            data = json.loads(line)
            chatglm_answer_list[data["label"].strip()] = data["predict"].strip()
    print(f"chatglm预测生成了{i}个答案")

    combined_answer = []
    example_dataset = json.load(open(original_file, "r", encoding="utf-8"))
    for key, data in tqdm(enumerate(example_dataset)):
        system = data["system"]
        instruction = data["instruction"]
        input = data["input"]
        history = data["history"]
        if data["output"].strip() in chatglm_answer_list:
            chatglm_answer = chatglm_answer_list[data["output"]].strip()
            combined_answer.append(
                {"system": system, "instruction": instruction, "input": input, "output": chatglm_answer, "history": history})
        else:
            # print("orignal answer")
            # print(data["output"])
            # print("==========" * 2)
            # print("chatglm answer")
            # print(chatglm_answer_list)
            # print("==========" * 2)
            continue
        # if key % 1000 == 0:
        #     # print(instruction)
        #     print("==========" * 10)
        #     print(key)
        #     print("==========" * 10)
        #     print(data["output"][:100])
        #     print("=========="*2)
        #     print(chatglm_answer[:100])
        # combined_answer.append({"instruction": instruction, "input": input, "output": chatglm_answer, "history": history})
    print(f"askbob_chatglm3共有{len(combined_answer)}个问题")
    return combined_answer

def concat_data_and_prediction_into_comparision(original_file, chatglm_file, is_train=True):
    cnt_drop = 0
    cnt_overall = 0
    cnt_zero = 0
    with open(chatglm_file, "r", encoding="utf-8") as f:
        chatglm_answer_list = {}
        for i, line in enumerate(f.readlines()):
            data = json.loads(line)
            chatglm_answer_list[data["label"].strip()] = data["predict"].strip()
    print(f"chatglm预测生成了{i}个答案")

    combined_answer = []
    example_dataset = json.load(open(original_file, "r", encoding="utf-8"))
    for key, data in tqdm(enumerate(example_dataset)):
        system = data["system"]
        instruction = data["instruction"]
        input = data["input"]
        history = data["history"]
        if data["output"].strip() in chatglm_answer_list:
            chatglm_answer = chatglm_answer_list[data["output"]].strip()
            combined_answer.append(
                {"system": system,
                 "instruction": instruction,
                 "input": input,
                 "output":[data["output"], chatglm_answer],
                 "history": history})
        else:
            # print("orignal answer")
            # print(data["output"])
            # print("==========" * 2)
            # print("chatglm answer")
            # print(chatglm_answer_list)
            # print("==========" * 2)
            continue
        # if key % 1000 == 0:
        #     # print(instruction)
        #     print("==========" * 10)
        #     print(key)
        #     print("==========" * 10)
        #     print(data["output"][:100])
        #     print("=========="*2)
        #     print(chatglm_answer[:100])
        # combined_answer.append({"instruction": instruction, "input": input, "output": chatglm_answer, "history": history})
    print(f"askbob_chatglm3共有{len(combined_answer)}个问题")
    return combined_answer


if __name__ == '__main__':
    # train_data = preprocess_jsonline(original_file="../../data/askbob_qa/train_askbob_0201_ft.json",
    #                                  chatglm_file="generated_predictions.jsonl",
    #                                  is_train=True)
    # json.dump(train_data, open("askbob_0201_chatglm3-vanilla.json", "w"), ensure_ascii=False, indent=4)

    train_data = concat_data_and_prediction_into_comparision(original_file="../../data/askbob_qa/train_askbob_0201_ft.json",
                                     chatglm_file="generated_predictions.jsonl",
                                     is_train=True)
    json.dump(train_data, open("askbob_0301_chatglm3-vanilla-glm4_comparision.json", "w"), ensure_ascii=False, indent=4)





