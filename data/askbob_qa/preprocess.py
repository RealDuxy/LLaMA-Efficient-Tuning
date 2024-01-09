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

slash = "\n"

def preprocess_jsonline(file_name):
    with open(file_name, "r", encoding="utf-8") as f:
        lines = []
        for line in f.readlines():
            data = json.loads(line)

            traning_prompt = data["traning_prompt"]
            question = data["gen_title"]
            query = traning_prompt.replace("{gen_title}", question)

            output = data["output"]
            contexts = data["contexts"]

            context_list = [f"【标题】：{context['title']}\n【内容】：{context['content']}" for context in contexts]
            context_str = "```\n\n```".join(context_list)

            prompt = template.replace("{question}", query).replace("{context}", context_str)

            lines.append({
                "system": system,
                'instruction': prompt,
                'input': "",
                'output': output,
                'history': []
            })
    return lines


def preprocess_llm_data(file_name):
    high_quality_datas = []
    normal_quality_datas = []
    askbob_datas = []
    with open(file_name, "r", encoding="utf-8") as f:
        data_list = json.load(f)

        for data in data_list:

            question_source = data["source_from"]
            query_for_training = data["query_for_training"]
            output = data["response"]
            contexts = data["all_contexts"]
            context_list = [f'''【标题】：{context['title']}\n【内容】：{context['content'].replace(slash, " ")}''' for context in contexts]
            random.shuffle(context_list)
            context_str = "```\n\n```".join(context_list)

            prompt = template.replace("{question}", query_for_training).replace("{context}", context_str)

            if question_source == "./dataset/rag_dataset.json":
                askbob_datas.append({
                    "system": system,
                    'instruction': prompt,
                    'input': "",
                    'output': output,
                    'history': []
                })
            elif question_source == "./dataset/1107_highQuality_gen.json":
                high_quality_datas.append({
                    "system": system,
                    'instruction': prompt,
                    'input': "",
                    'output': output,
                    'history': []
                })
            elif question_source == "./dataset/1107_normal_gen.json":
                normal_quality_datas.append({
                    "system": system,
                    'instruction': prompt,
                    'input': "",
                    'output': output,
                    'history': []
                })
    return askbob_datas, high_quality_datas, normal_quality_datas

if __name__ == '__main__':
    # train_data = preprocess_jsonline("1115_highQuality.json")
    # json.dump(train_data, open("train.json", "w"), ensure_ascii=False, indent=4)
    #
    # eval_data = preprocess_jsonline("1115_highQuality_eval.json")
    # json.dump(eval_data, open("dev.json", "w"), ensure_ascii=False, indent=4)

    askbob_datas, high_quality_datas, normal_quality_datas = preprocess_llm_data("llm_data.json")
    json.dump(askbob_datas, open("train_askbob.json", "w"), ensure_ascii=False, indent=4)
    json.dump(high_quality_datas, open("train_high_quality.json", "w"), ensure_ascii=False, indent=4)
    json.dump(normal_quality_datas, open("train_normal.json", "w"), ensure_ascii=False, indent=4)
