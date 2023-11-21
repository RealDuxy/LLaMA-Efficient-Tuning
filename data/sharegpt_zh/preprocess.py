import json
import random

from tqdm import tqdm
from transformers import AutoTokenizer

template = """```
{context}
```

{question}"""

system = """你是就职于中国平安人寿保险公司的保险专家，你的名字叫做安安。你需要使用由三个`符号包括的内容（即检索资料）回答问题或者执行指令。
如果检索资料内存在与问题几乎相同的标题，你需要复制检索资料内容的原文进行回答。
如果检索资料内存在与问题非常相似的标题，你需要模仿检索资料的内容的结构，使用相关内容的原文进行回答。
如果检索资料内不存在相同或相似的标题，但是内容里提供了足够回答问题所需要的知识，你需要有选择的选取相关的内容归纳总结，并回答问题。
如果检索资料无法提供足够回答问题所需要的知识，你需要拒绝回答，并明确指出你缺少哪些知识。
你必须保证你的回答里的所有内容都有检索资料的支持，因此你只能使用给定的检索资料进行回答，不允许进行额外的推理、猜测、延伸、想象。"""

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
            chatglm_answer_list[data["label"][:100]] = data["predict"]
    print(f"chatglm预测生成了{i}个答案")

    combined_answer = []
    example_dataset = json.load(open(original_file, "r", encoding="utf-8"))
    for key, data in tqdm(enumerate(example_dataset)):
        instruction = data["instruction"]
        input = data["input"]
        history = data["history"]
        if data["output"][:100] in chatglm_answer_list:
            chatglm_answer = chatglm_answer_list[data["output"][:100]]
            combined_answer.append(
                {"instruction": instruction, "input": input, "output": chatglm_answer, "history": history})
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
    print(f"sharegpt_chatglm3共有{len(combined_answer)}个问题")
    return combined_answer

if __name__ == '__main__':
    train_data = preprocess_jsonline(original_file="../sharegpt_zh_27k.json",
                                     chatglm_file="generated_predictions.jsonl",
                                     is_train=True)
    json.dump(train_data, open("../sharegpt_chatglm3_zh_27k.json", "w"), ensure_ascii=False, indent=4)



