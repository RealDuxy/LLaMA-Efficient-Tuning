import json
import random



template = """Here's a Human's prompt and AI assistant's output:
```
Human: 
{prompt}

AI assistant: 
{output}
```

You task is to generate a more accurate, detailed and complex prompt to replace the human's prompt based on the conversation they had. 
The new prompt should be more accurate, and detailed than the current one and it can not be longer than the output. 
You should directly generate the new prompt. 
The new prompt:"""


def preprocess_jsonline(original_file, chatglm_file, is_train=True):
    cnt_drop = 0
    cnt_overall = 0
    cnt_zero = 0
    with open(chatglm_file, "r", encoding="utf-8") as f:
        chatglm_answer_list = []
        for i, line in enumerate(f.readlines()):
            data = json.loads(line)
            chatglm_answer_list.append(data["predict"])
    print(f"chatglm预测生成了{i}个答案")

    combined_answer = []
    example_dataset = json.load(open(original_file, "r", encoding="utf-8"))
    for key, data in enumerate(example_dataset):
        instruction = data["instruction"]
        input = data["input"]
        # answer_gpt4 = data["output"]

        combined_answer.append({"instruction": instruction, "input": input, "output": chatglm_answer_list[key]})
    print(f"alpaca共有{key}个问题")
    return combined_answer

def regenerate_prompt_for_alpaca(original_file):
    combined_answer = []
    example_dataset = json.load(open(original_file, "r", encoding="utf-8"))
    for key, data in enumerate(example_dataset):
        instruction = data["instruction"]
        input = data["input"]
        output = data["output"]
        prompt = instruction + "\n" + input
        input = {"prompt": prompt, "output": output}
        regenerate_prompt = template.format(**input)
        combined_answer.append({"prompt": regenerate_prompt, "output": ""})
    print(f"all data: {len(combined_answer)}")
    json.dump(combined_answer, open("alpaca_new_prompt.json", "w"), ensure_ascii=False, indent=4)

if __name__ == '__main__':
    # train_data = preprocess_jsonline(original_file="../alpaca_gpt4_data_zh.json",
    #                                  chatglm_file="generated_predictions.jsonl",
    #                                  is_train=True)
    # json.dump(train_data, open("../alpaca_chatglm3_data_zh.json", "w"), ensure_ascii=False, indent=4)

    regenerate_prompt_for_alpaca(original_file="../alpaca_gpt4_data_en.json")



