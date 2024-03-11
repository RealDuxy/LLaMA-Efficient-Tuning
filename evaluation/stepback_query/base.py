# -*- encoding: utf-8 -*-
"""
@File    : base.py
@Time    : 27/2/2024 15:39
@Author  : Duxy
@Email   : du.xi.yang@qq.com
@Software: PyCharm
"""
import json
from concurrent.futures.thread import ThreadPoolExecutor
from copy import deepcopy

from tenacity import wait_random_exponential, retry, stop_after_attempt
from typing import List

from tqdm import tqdm


class BaseAgent:
    def __init__(self, template_file: str, model_invoke):
        self.templates = json.load(open(template_file, "r", encoding="utf-8"))
        self.model_invoke = model_invoke

    def assemble_messages(self, **kwargs):
        chat_templates = deepcopy(self.templates)

        user_prompt = chat_templates["prompt"]
        messages = chat_templates["history"]

        for key, value in kwargs.items():
            temp_value = value
            temp_key = "{" + key + "}"
            if temp_key not in user_prompt:
                print(f"input_key: {temp_key} not in prompt, check you keys and template")
                continue
            user_prompt = user_prompt.replace(temp_key, temp_value)

        messages.append({"role": "user", "content": user_prompt})

        return messages

    def invoke(self, **kwargs):
        messages = self.assemble_messages(**kwargs)
        response = self.model_invoke(messages)
        return response

    def para_invoke(self, input_kwargs, max_workers=4):
        executer = ThreadPoolExecutor(max_workers=max_workers)
        results = []
        for i, result in enumerate(executer.map(lambda x: self.invoke(**x), input_kwargs)):
            results.append(result)
        return results

    def save_results(self, results: List[str], save_file):
        with open(save_file, "a") as wf:
            wf.write("\n".join(results))

    def batch_invoke_and_save(self, save_file, batch_size=20, save_step=5, **kwargs):

        data_num = len(list(kwargs.values())[0])
        input_kwargs = []
        for i in range(data_num):
            input = {}
            for key, values in kwargs.items():
                input[key] = values[i]
            input_kwargs.append(input)

        with open(save_file, "r") as f:
            results = f.readlines()

        print(f"当前已经跑了{len(results)}条，继续开始")

        for i in tqdm(range(len(results), data_num, batch_size)):
            results += self.para_invoke(input_kwargs[i: i+batch_size])
            if i % save_step == 0:
                self.save_results(results, save_file)
                print(f"当前已经跑了{len(results)}条，保存至{save_file}")

        self.save_results(results, save_file)
        print(f"全部完成，共计{len(results)}条，最后保存至{save_file}")
        return results









if __name__ == '__main__':
    system = "你是一个文本修改专家，你的任务是根据给定的检索资料、用户提问和原始答案以及" \
             "评判老师的打分标准和原始答案打分意见，生成一个修改后的答案，修改后的答案应该比输入的文本获得更高的分数。" \
             "请只输出修改后的答案，并且保持与原始答案的语义一致，不要输出其他内容。"
    prompt_template = "# 检索资料 \n {refs}\n" \
             "# 用户提问 \n {questions}\n" \
             "# 原始答案 \n {original_answer}\n" \
             "# 老师打分标准 \n {standards}\n" \
             "# 老师打分意见\n {reasons}\n" \
             "请直接输出修改后的答案，并且尽可能不修改原始答案的文本，不要输出任何其他内容。"

    rewrite_experts = BaseAgent(system, prompt_template, ["refs", "questions", "original_answer", "standards", "reasons"])

    refs = "```\n【标题】：让自家亲戚买个百万医疗都费劲，什么都问还不买，有什么好的办法吗\n【内容】：非常理解您的困扰。在向亲戚推销保险时，有几个方法可能会有所帮助：\n\n1. 了解需求：首先，了解亲戚的具体需求和担忧。询问他们是否有关于医疗保险的具体问题或疑虑，以便您能够提供相关的信息和解答。\n\n2. 提供专业建议：作为保险销售专家，您可以提供专业的建议，解释医疗保险的重要性和优势。强调医疗保险可以为他们提供经济保障，帮助应对突发的医疗费用。\n\n3. 说明案例：分享一些真实的案例，说明医疗保险的好处。这些案例可以是您亲自经历的，或者是其他人的故事，以便让亲戚更好地理解医疗保险的重要性。\n\n4. 解答疑问：耐心回答亲戚的问题，并提供清晰的解释。确保他们理解保险的条款、保障范围、理赔流程等关键信息。\n\n5. 提供比较分析：如果亲戚对比不同的医疗保险产品，您可以提供详细的比较分析，帮助他们选择最适合的保险方案。\n\n6. 引用权威资料：引用权威机构或专家的研究和报告，以支持您的观点。这样可以增加亲戚对医疗保险的信任和认可。\n\n最重要的是，保持耐心和尊重，尊重亲戚的决定，并尊重他们可能有的疑虑和担忧。通过提供专业的建议和信息，您可以帮助他们做出明智的决策。\n```"
    questions = "让自家亲戚买个百万医疗都费劲，什么都问还不买，有什么好的办法吗"
    original_answer = """非常理解您的困扰。在向亲戚推销保险时，有几个方法可能会有所帮助：

1. 了解需求：首先，了解亲戚的具体需求和担忧。询问他们是否有关于医疗保险的具体问题或疑虑，以便您能够提供相关的信息和解答。

2. 提供专业建议：作为保险销售专家，您可以提供专业的建议，解释医疗保险的重要性和优势。强调医疗保险可以为他们提供经济保障，帮助应对突发的医疗费用。

3. 说明案例：分享一些真实的案例，说明医疗保险的好处。这些案例可以是您亲自经历的，或者是其他人的故事，以便让亲戚更好地理解医疗保险的重要性。

4. 解答疑问：耐心回答亲戚的问题，并提供清晰的解释。确保他们理解保险的条款、保障范围、理赔流程等关键信息。

5. 提供比较分析：如果亲戚对比不同的医疗保险产品，您可以提供详细的比较分析，帮助他们选择最适合的保险方案。

6. 引用权威资料：引用权威机构或专家的研究和报告，以支持您的观点。这样可以增加亲戚对医疗保险的信任和认可。

最重要的是，保持耐心和尊重，尊重亲戚的决定，并尊重他们可能有的疑虑和担忧。通过提供专业的建议和信息，您可以帮助他们做出明智的决策。"""
    reasons = '''"ANSWER_RELEVANCE_score": 0.0, "ANSWER_RELEVANCE_reason": "学生的回答直接重复了题目提供的第一段背景材料中的内容，没有给出针对问题“有什么好的办法吗”自己的解答，因此在内容的完整性上存在严重缺陷。根据评分标准，该回答没有涉及问题的解决方案，所以基本上没回答问题。", "FACTUAL_CORRECTNESS_score": 1.0, "FACTUAL_CORRECTNESS_reason": "学生的回答复述了背景材料，背景材料的内容认定为客观事实，因此学生回答具有事实性。", "FAITHFULNESS_score": 1.0, "FAITHFULNESS_reason": "学生的回答与背景资料中的内容一致，没有与参考材料产生冲突或矛盾，因此符合忠实度评价标准。", "FLUENCY_score": 3.0, "FLUENCY_reason": "学生的回答摘抄自参考材料，材料本身在语法、词汇和短语选择上是流畅的，所以回答在流畅性方面没有问题。"'''

    modified_answer = rewrite_experts.invoke(
        refs = refs,
        questions = questions,
        original_answer = original_answer,
        standards = standards,
        reasons=reasons
    )

    print(modified_answer)

