import json
import random

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

def preprocess_jsonline(file_name, is_train=True):
    cnt_drop = 0
    cnt_overall = 0
    cnt_zero = 0
    with open(file_name, "r", encoding="utf-8") as f:
        lines = []
        for line in json.load(f):
            cnt_overall += 1
            data = line
            question_list = data["问题泛化"].split(";")
            query = random.choice(question_list).strip()

            output = data["最终答案"]

            contexts = []
            accumalated_length = 0
            for key in ["给GPT的SourceA1","给GPT的SourceB1","给GPT的SourceA2","给GPT的SourceB2"]:
                if data[key].strip() == "": continue
                source_len = len(tokenizer(data[key]).input_ids)
                accumalated_length += source_len

            if 0 < accumalated_length <= 1400:
                for key in ["给GPT的SourceA1", "给GPT的SourceB1", "给GPT的SourceA2", "给GPT的SourceB2"]:
                    if data[key] == "": continue
                    contexts.append({"content": data[key]})
            elif accumalated_length == 0:
                cnt_zero += 1
                if is_train:
                    continue
            else:
                print(f"question：{query}")
                print(f"长度：{accumalated_length}")
                cnt_drop += 1
                if is_train:
                    continue

            context_list = [f"【产品相关条款内容】：{context['content']}" for context in contexts]
            context_str = "```\n\n```".join(context_list)
            if len(context_str) <= 20:
                print(query)

            prompt = template.replace("{question}", query).replace("{context}", context_str)
            lines.append({
                "system": system,
                'instruction': prompt,
                'input': "",
                'output': output,
                'history': []
            })

    print(cnt_overall)
    print(cnt_drop)
    print(cnt_zero)
    print()
    return lines

if __name__ == '__main__':
    train_data = preprocess_jsonline("产品问答体系-训练集.json")
    json.dump(train_data, open("train.json", "w"), ensure_ascii=False, indent=4)

    eval_data = preprocess_jsonline("产品问答体系-评估集.json", False)
    json.dump(eval_data, open("dev.json", "w"), ensure_ascii=False, indent=4)

