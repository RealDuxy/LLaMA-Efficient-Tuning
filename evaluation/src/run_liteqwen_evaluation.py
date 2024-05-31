# -*- encoding: utf-8 -*-
"""
@File    : rag.py
@Time    : 15/5/2024 12:25
@Author  : Duxy
@Email   : du.xi.yang@qq.com
@Software: PyCharm
"""

import json
import os.path
import re
import sys
import time
from random import shuffle

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from base import BaseLiteLLMAgent
from utils import get_qwen_response, batch_dataset_iterator, get_chatglm_response

model_adapter_name_map = {
    "chatglm": "",
    "chatglm-rag-0515": "default",
    "chatglm-rag-0515-dpo": "align",
    "qwen": "",
    "qwen-rag-0527-exp2": "default",
    "qwen-rag-0529-simpo-exp2": "rag1",
    "qwen-rag-0529-simpo-exp3": "rag2"
}

test_case = """【标题】：虽然客户具备购买力，但对保险产生了疑虑，应该采取何种措施？
【内容】：亲爱的同仁，面对客户不认可保险的情况，我非常理解你的困惑。这是我们销售过程中经常会遇到的问题，但请记住，每一个挑战都是一个机会。下面我会结合公司的情况、专业服务体系和销售流程与技巧，为你提供一个详细的解答。
1. 问题总体分析：
首先，我们要明白“不太认可保险”背后的真正原因。这可能是由于客户对保险的认知不足、过去的不良经验、或对投资收益的误解等因素导致的。核心问题在于信任和认知的缺失。
2. 针对性分析：
人群：客户有购买能力，说明他们有对未来进行风险管理和投资的需求。
场景：面对这样的客户，我们更应该从教育和咨询的角度进行沟通，而不仅仅是销售。
情况：客户不认可保险，这也许是因为他们之前没有得到过真正的专业建议，或者他们对保险的了解都是基于市场上的负面新闻。
3. 结合公司情况与销售流程技巧进行解答：
利用平安的荣誉与服务：中国平安是国内乃至全球的顶尖保险企业，我们有着丰富的产品线和服务体系。“one ping an”生态版图就是我们的优势。在与客户沟通时，强调我们与医疗机构、药店、体检中心等的合作关系，展示我们在“保险+健康管理”、“保险+高端康养”等方面的专业能力。
教育客户：针对客户的疑虑，为他们提供保险知识，解释保险的真正意义，以及它如何帮助他们在面对意外、疾病等风险时得到保障。可以通过真实的案例来讲述，使之更具有说服力。
建立信任：从了解客户开始，深入挖掘他们的需求。例如，他们关心的是家庭保障、资产保值增值还是退休计划。结合他们的实际情况，为他们制定专业的保险方案。
细心倾听：在与客户交谈时，耐心听取他们的疑虑和担忧，并针对性地提供解答。
后续服务：在客户购买产品后，不要忘记提供持续的服务，如定期的保单回访、健康提醒等，让客户真正感受到平安的专业和用心。
4. 结论：
面对不认可保险的客户，我们的目标不仅是完成销售，更重要的是帮助他们认识到保险的价值，建立起真正的信任。只有这样，我们才能真正做到客户为中心，为他们提供真正有价值的服务。希望我的建议能帮到你，一起加油！

【标题】：尽管客户财力雄厚，但对保险产生疑虑，应该如何解决？
【内容】：问题分析：
首先，我们来分析这个问题。客户有购买能力，这意味着他们有经济实力为自己或家人购买保险。但关键在于他们“不太认可保险”。这可能是因为他们对保险缺乏了解、过去有过不良经验、听说过负面口碑或者对保险的价值和回报持怀疑态度。
针对性分析：
人群：经济实力较强的客户，这类客户对金钱的价值有深入的认识，希望其投资可以带来更好的回报。
场景：可能是在与客户的沟通中，或在听到他们的反馈时，得知他们对保险持怀疑态度。
情况：客户可能由于缺乏对保险的了解，或曾经听说过不良的经验或者对保险的长期回报不持乐观态度。
结合公司情况、专业服务体系和销售流程与技巧的解答：
提供教育和宣传：首先，要了解客户为什么对保险持有疑虑。是因为他们不了解保险，还是因为他们曾经有过不良经验？一旦了解了原因，你可以提供相应的教育材料和宣传，如平安的荣誉、专业服务体系等，来帮助他们更好地了解和信任保险。
分享成功案例：可以分享中国平安的成功案例和客户的真实故事，让他们看到其他人如何通过购买保险获得保障和利益。
一对一咨询：针对他们的疑虑，为他们提供一对一的咨询服务。通过深入交流，找出他们对保险的疑虑和担忧，并针对性地解答。
介绍专业服务体系：向他们详细介绍“ONE PING AN”生态版图，如“保险+健康管理”、“保险+居家养老”等专业服务模式，展示平安保险不仅是一家保险公司，而是一个提供综合健康和养老服务的综合平台。
建立长期关系：不要急于让客户购买，而是先建立起信任关系。在销售流程中，与客户进行长期的互动交流，了解他们的需求，再提供合适的保险方案。
利用销售技巧：在面对这种客户时，可以运用你所学的销售技巧，如激发客户需求、解决客户异议等，帮助他们看到购买保险的价值。
总的来说，针对这类客户，最重要的是建立起信任关系，让他们了解和认可保险的价值。只有这样，他们才会放心地为自己和家人购买保险。

【标题】：客户反应咱的产品附加险太多怎么解决？
【内容】：客户反应咱的产品附加险太多怎么解决？

1. 首先，我完全理解客户对于产品附加险种类繁多的疑虑。我们平安寿险的产品附加险种类确实很多，这是为了满足不同客户的个性化需求而设计的。

2. 产品附加险的多样性是我们平安寿险的优势之一。我们致力于开发各种不同的附加险，以满足客户在家族遗传疾病史等方面的不同需求。

3. 客户可以根据自己的家族状况和个人需求选择适合自己的附加险。这样的选择权对于客户来说非常重要，因为每个家庭的情况都不同，只有有多样的选择，才能更好地满足客户的需求。

4. 就像去商店购买商品一样，客户希望商店的品种齐全，这样才能找到适合自己的产品。如果一个商店的品种单一，无法满足客户的需求，客户可能会选择去其他商店购买所需的产品。

综上所述，客户对于产品附加险种类繁多的疑虑是可以理解的。然而，我们平安寿险提供多样的附加险种类，旨在满足不同客户的个性化需求。客户可以根据自己的家族状况和个人需求选择适合自己的附加险，这样的选择权对于客户来说非常重要。我们相信，多样的选择能够更好地满足客户的需求，提供更全面的保障。

【标题】：客户不太喜欢附加保险，觉得没啥用，怎么跟客户说
【内容】：在这个问题中，我们面临的主要挑战是如何处理客户对赠险（通常是保险公司为了促销而提供的免费附加保险）的负面态度。他们可能认为这些赠险没有实际价值或与他们的需求不符。以下是对这个问题的分析和建议策略：
问题核心要点分析：
客户的抵触可能源于对赠险价值的误解或不了解。
客户可能对“免费”感到怀疑，认为这背后有其他附加条件或认为服务质量不高。
与客户沟通时，重点是教育和引导，而不是强推产品。
针对性分析：
人群：这些客户可能是经验丰富的保险购买者，他们倾向于谨慎对待保险产品，尤其是看似“附加”的条款。
场景：在推销过程中，当代理人提及赠险时，客户可能立即表现出抵触情绪。
情况：客户的反应可能基于之前的不良经历（如赠险索赔时遇到问题）或是对保险的误解。
结合公司情况、专业服务体系和销售技巧的解答：
首先，了解和重申平安保险的品牌价值和对客户承诺的重视。强调公司的信誉和我们提供的全面专业服务。
其次，不要立即反驳客户的观点，而是要倾听并理解他们的顾虑。从客户的角度出发，解释赠险的好处，以及它是如何增强他们现有保险计划的补充。
展示赠险的实际案例和成功索赔的故事，以便客户了解它的实际应用和好处。如果可能的话，分享其他客户的积极反馈。
强调赠险的无条件性。确保客户理解没有隐藏的费用或条款，赠险是对他们的额外保障，旨在增强他们的满意度和忠诚度。
如果客户依然持怀疑态度，可以考虑展示如何将赠险与他们的主要保单相结合，以提供更全面的保障。重点关注赠险如何填补现有计划的潜在缺口。
具体、实际的解决方案：
使用具体的数据和案例研究来证明赠险的价值。例如，展示过去一年中，有多少客户实际利用了赠险进行了成功的索赔。
提供定制化的建议，说明如果在特定情况下（例如交通事故、家庭突发事件等）赠险如何提供额外保障。
如果客户还是不感兴趣，不要强迫他们接受。相反，注重维护与客户的关系，表明你尊重他们的决定，并且在他们需要时随时为他们提供服务。
通过上述方法，代理人可以更有效地与客户沟通赠险的价值，转变客户的看法，并在整个过程中建立信任。这种方法既体现了平安保险对客户需求的关注，也展示了公司作为保险行业领导者的专业性和可靠性。

【标题】：对于客户不喜欢的附赠保险，应该有何应对之策？
【内容】：分析
总体分析：
客户对领赠险的抵触心理可能源于对该产品的不了解、对保险产业的误解、或曾经的不良购买经验。领赠险是一种赠送给客户的保险产品，但如果客户不愿接受，需要我们深入了解原因，并通过合适的策略转化其态度。
人群、场景、情况分析：
人群：不愿意领取赠险的客户可能是对保险产品缺乏信任的人群，或者是对自己不需要的产品抱有疑虑的客户。
场景：在推荐赠险时，可能因为销售手法、语言表述、或者环境因素导致客户的反感。
情况：客户可能认为领赠险之后会被骚扰、或者担心领赠险后会有其他隐性消费。
公司情况、专业服务体系、销售流程与技巧：
中国平安作为国内顶尖的保险公司，其产品都是经过严格筛选与测试的，具有很高的品质保证。其“ONE PING AN”生态版图和各种综合服务体系都是为了提供更好的客户体验。
解答
了解客户的顾虑：首先，要与客户进行深入的沟通，了解他们为什么不喜欢或不愿意领取赠险。是对产品的疑虑、还是对销售方式的不满？
提供真实、透明的产品信息：可以详细向客户介绍赠险的特点、覆盖范围、保障期限等，确保客户对赠险有全面的了解。并强调中国平安对客户的承诺，不会有任何隐性消费或未经同意的推销行为。
展示公司优势：介绍中国平安的品牌、荣誉、及“ONE PING AN”生态版图的优势。让客户理解，赠险只是为了让他们体验到平安的服务，而不是带来任何麻烦。
结合销售技巧：在与客户沟通时，可以采用“疑问-解答”模式，针对客户的疑虑提供有针对性的解答，使沟通更为流畅。另外，也可以分享一些真实的赠险使用案例，使客户对赠险有更为直观的认识。
后续跟进：即使客户此次没有领取赠险，也可以继续保持与他们的联系。在未来，客户可能因为其他需求再次考虑保险产品，这时，过去的良好沟通和服务就会转化为你的优势。
总之，面对客户的反馈或抵触，关键是以真诚的态度、专业的知识和技能来建立信任和关系，进而转化为销售机会。"""

def run_rag_evaluation(data_dir, output_dir,
                       template_file,
                       model_name="names",
                       max_samples=None,
                       model_invoke=get_qwen_response):
    rag_agent = BaseLiteLLMAgent(template_file=template_file, model_invoke=model_invoke)

    for data_file in os.listdir(data_dir):
        output_file = data_file.replace(".json", f"_output.json")
        data_file = os.path.join(data_dir, data_file)
        model_output_dir = output_dir + f"/{model_name}/"
        os.makedirs(model_output_dir, exist_ok=True)
        output_file = os.path.join(model_output_dir, output_file)
        print(f"Processing data file: {data_file}")
        results = []
        for datas in tqdm(batch_dataset_iterator(data_file, batch_size=4, max_samples=max_samples)):
            predictions = rag_agent.para_invoke(adapter_name=[model_name] * len(datas["question"]),
                                                **{"question": datas["question"]
                                                    , "requirement": datas["requirement"]
                                                    , "context": datas["context"]})
            datas.update({"pred": predictions})
            for i in range(len(predictions)):
                results.append({
                    "question": datas["question"][i],
                    "requirement": datas["requirement"][i],
                    "context": datas["context"][i],
                    "output": datas["output"][i],
                    "pred": datas["pred"][i]
                })

        # results 保存到output_file
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"Results saved to {output_file}")


def run_rag_all_negative_rejection_answer(data_dir, output_dir,
                                          template_file,
                                          model_name="names",
                                          max_samples=None,
                                          model_invoke=get_chatglm_response):
    rag_agent = BaseLiteLLMAgent(template_file=template_file, model_invoke=model_invoke)

    for data_file in os.listdir(data_dir):
        output_file = data_file.replace(".json", f"_output.json")
        data_file = os.path.join(data_dir, data_file)
        model_output_dir = output_dir + f"/{model_name}/"
        os.makedirs(model_output_dir, exist_ok=True)
        output_file = os.path.join(model_output_dir, output_file)
        print(f"Processing data file: {data_file}")
        results = []
        for datas in tqdm(batch_dataset_iterator(data_file, batch_size=4, max_samples=max_samples)):
            predictions = rag_agent.para_invoke(adapter_name=[model_name] * len(datas["question"]),
                                                **{"question": datas["question"]
                                                    , "requirement": datas["requirement"]
                                                    , "context": datas["context"]})
            datas.update({"pred": predictions})
            for i in range(len(predictions)):
                results.append({
                    "question": datas["question"][i],
                    "requirement": datas["requirement"][i],
                    "context": datas["context"][i],
                    "output": datas["output"][i],
                    "pred": datas["pred"][i]
                })

        # results 保存到output_file
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"Results saved to {output_file}")


def run_rag_prediction(data_dir,
                       output_dir,
                       template_file,
                       model_name="names",
                       max_samples=None,
                       model_invoke=get_qwen_response,
                       sorted_by_output=False):

    def shuffle_context(context_str):
        context_list = context_str.split("【标题】：")

        shuffle(context_list)
        return "【标题】：" + "\n\n【标题】：".join(context_list)

    os.makedirs(output_dir, exist_ok=True)

    rag_agent = BaseLiteLLMAgent(template_file=template_file, model_invoke=model_invoke)


    for data_file in os.listdir(data_dir):
        time_start = time.time()
        # 过滤文件
        if "train_0524" not in data_file: continue

        output_file = data_file.replace(".json", f"_output.json")
        data_file = os.path.join(data_dir, data_file)
        model_output_dir = output_dir + f"/{model_name}/"
        os.makedirs(model_output_dir, exist_ok=True)
        output_file = os.path.join(model_output_dir, output_file)
        print(f"Processing data file: {data_file}")
        results = []
        for i, datas in tqdm(
                enumerate(batch_dataset_iterator(data_file, batch_size=4, max_samples=max_samples,sorted_by_output=sorted_by_output)),
                desc=f"processing {data_file.split('/')[-1]}"):
            # shuffled_contexts = [shuffle_context(x) for x in datas["context"]]
            predictions = rag_agent.para_invoke(adapter_name=[model_adapter_name_map[model_name]] * len(datas["question"]),
                                                **{"question": datas["question"]
                                                    , "requirement": datas["requirement"]
                                                    , "context": datas["context"]})
            datas.update({"pred": predictions})
            for j in range(len(predictions)):
                results.append({
                    "question": datas["question"][j],
                    "requirement": datas["requirement"][j],
                    "context": datas["context"][j],
                    "output": datas["output"][j],
                    "pred": datas["pred"][j]
                })
            if i % 200 == 1:
                # results 保存到output_file
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=4)
                print(f"question:{results[-1]['question']}")
                print(f"output:{results[-1]['output']}")
                print(f"pred:{results[-1]['pred']}")
                print(f"temp {i} Results saved to {output_file}")

        # results 保存到output_file
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"Results saved to {output_file}")
        time_end = time.time()
        print(f"{model_name}-{data_file.split('/')[-1]} time cost: {(time_end - time_start)}")
        print(f"{model_name}-{data_file.split('/')[-1]} time cost per cost: {(time_end - time_start) / (4*(i+1))}")
        print(f"{model_name}-{data_file.split('/')[-1]} time cost per batch: {(time_end - time_start) / (i+1)}")

def run_rag_comparison(data_dir,
                       output_dir,
                       template_file,
                       model_names=["qwen"],
                       max_samples=None,
                       model_invoke=get_qwen_response,
                       sorted_by_output=False):

    os.makedirs(output_dir, exist_ok=True)

    rag_agent = BaseLiteLLMAgent(template_file=template_file, model_invoke=model_invoke)

    for data_file in os.listdir(data_dir):
        time_start = time.time()
        output_file = data_file.replace(".json", f"_output.json")
        data_file = os.path.join(data_dir, data_file)
        model_output_dir = output_dir + f"/0527_comparison/"
        os.makedirs(model_output_dir, exist_ok=True)
        output_file = os.path.join(model_output_dir, output_file)
        print(f"Processing data file: {data_file}")
        results = []
        for i, datas in tqdm(
                enumerate(batch_dataset_iterator(data_file, batch_size=4, max_samples=max_samples,sorted_by_output=sorted_by_output)),
                desc=f"processing {data_file.split('/')[-1]}"):
            # shuffled_contexts = [shuffle_context(x) for x in datas["context"]]
            for model_name in model_names:
                predictions = rag_agent.para_invoke(adapter_name=[model_adapter_name_map[model_name]] * len(datas["question"]),
                                                    **{"question": datas["question"]
                                                        , "requirement": datas["requirement"]
                                                        , "context": datas["context"]})
                print(f"========{model_name}_prediction finished============")
                print(f"{predictions[0]}")
                datas.update({f"{model_name}_prediction": predictions})

            for j in range(len(predictions)):
                tmp_result = {
                    "question": datas["question"][j],
                    "requirement": datas["requirement"][j],
                    "context": datas["context"][j],
                    "output": datas["output"][j]
                }
                for model_name in model_names:
                    tmp_result.update({f"{model_name}_prediction": datas[f"{model_name}_prediction"][j]})
                results.append(tmp_result)

            if i % 5 == 1:
                # results 保存到output_file
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=4)
                print(f"question:{results[-1]['question']}")
                print(f"output:{results[-1]['output']}")
                for model_name in model_names:
                    print({f"{model_name}_prediction": results[-1][f"{model_name}_prediction"]})
                print(f"temp {i} Results saved to {output_file}")

        # results 保存到output_file
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"Results saved to {output_file}")
        time_end = time.time()
        print(f"{data_file.split('/')[-1]} time cost: {(time_end - time_start)}")
        print(f"{data_file.split('/')[-1]} time cost per cost: {(time_end - time_start) / (4*(i+1))}")
        print(f"{data_file.split('/')[-1]} time cost per batch: {(time_end - time_start) / (i+1)}")

def test_single_agent(model_name="qwen-rag-0529-simpo-exp2"):
    adapter_name = model_adapter_name_map[model_name]
    requirement = "1. 请考虑客户的具体情况和个性化需求，提供定制化的解决方案。2. 在解答中，不仅要关注产品本身，还要考虑销售流程和技巧的改进。3. 尽量避免使用过于笼统或模糊的表述，而是要具体、实际。4. 在解答中，要充分展示公司的优势和服务体系。"
    context = test_case
    question = "客户不喜欢我给设计的产品组合中的附加险，该怎么应对？"
    template_file = "template/template.json"
    model_invoke = get_qwen_response
    # debug_agent = BaseLiteLLMAgent(template_file="template/template_debug.json", model_invoke=get_qwen_response)
    rag_agent = BaseLiteLLMAgent(template_file=template_file, model_invoke=get_chatglm_response)
    # predictions = debug_agent.invoke(adapter_name=model_adapter_name_map[model_name],
    #                                     **{"question": question})
    # print(predictions)
    predictions = rag_agent.invoke(adapter_name=model_adapter_name_map[model_name],
                                        **{"question": question
                                            , "requirement": requirement
                                            , "context": context})
    print(predictions)

if __name__ == '__main__':
    test_single_agent("default")
    # time_start = time.time()
    # run_rag_comparison(
    #     data_dir="dataset/evaluation_dataset",
    #     output_dir="output/evaluation_dataset",
    #     template_file="template/template.json",
    #     model_names=["qwen","qwen-rag-0527-exp2","qwen-rag-0529-simpo-exp2","qwen-rag-0529-simpo-exp3"],
    #     max_samples=4,
    #     model_invoke=get_qwen_response
    # )
    # time_end = time.time()
    # print(f"total time cost: {(time_end - time_start)}")
