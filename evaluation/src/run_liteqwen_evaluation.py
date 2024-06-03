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
    "qwen-rag-0601-simpo-exp1": "rag1",
    "qwen-rag-0601-dpo-exp1": "rag2"
}

test_case = {
        "question": "客户觉得有了社保就不需要商业保险，怎么处理呢",
        "contexts": "【标题】：如何找出有潜力的客户？\n【内容】：感谢您的提问，我们来逐步分析并尝试回答这个问题。1. 分析开门红的特点首先，开门红是保险公司每年的一项重要策略，目的是吸引更多客户通过推出一系列有限时优惠的产品。开门红的产品主要为财富类保险产品，如增额终身寿、年金险等，这些产品的卖点主要是储蓄、理财、高收益。2. 梳理开门红客户的主要步骤目标群体识别：确定您的目标客户群体。考虑那些对于理财、养老、教育、传承和保障有需求的人。例如，中年人士可能对于养老和教育有更大的需求，而年轻家庭可能更加关注传承和保障。分析现有客户数据库：根据您已有的客户信息，筛选出可能对开门红产品感兴趣的客户。可以基于他们之前的购买记录、查询记录或者您与他们的沟通记录来进行筛选。了解客户的需求：通过电话或面对面交流了解他们的保险需求。例如，询问他们是否对于教育、养老或理财有任何计划和需求。推介相应的产品：根据客户的需求，向他们推介相应的开门红产品。利用之前提供的话术，强调产品的稳定收益、资产配置、长期储蓄、保障功能和未来规划的优势。组织产说会：考虑在开门红期间组织产品推介销售说明会。邀请您的目标客户参加，利用公司整体力量和现场氛围帮助促成签单。持续跟进：与客户保持良好的沟通，定期了解他们的需求变化，并提供相关的产品更新和优惠信息。3. 注意事项避免一刀切：每个客户的需求都是独特的，所以需要根据他们的具体情况进行个性化的推介。确保诚信：在推介产品时，确保提供真实、完整的信息，不要为了销售而过度夸大产品的优点。开门红期间的密集性：由于开门红是一个短期内的促销活动，所以代理人需要确保在这个期间与客户的沟通更加密集和高效。综上所述，梳理开门红客户需要有针对性的策略，结合客户的实际需求推介合适的产品，并确保与客户的持续沟通和服务。希望以上建议能够帮助您有效地梳理开门红客户。\n\n【标题】：请问大家都在哪里搜寻打开财富之门的潜在客户呢？\n【内容】：感谢您的提问，我们来逐步分析并尝试回答这个问题。1. 分析开门红的目标客户群体客户画像：首先，需要明确开门红产品的目标客户是谁。考虑到产品特点，目标客户可能包括有一定储蓄能力的中产阶级、关注养老规划的中老年人、对子女教育规划感兴趣的年轻家长等。2. 寻找客户的渠道现有客户推荐：利用现有客户网络。满意的客户是最好的推荐人。可以提供一定的激励措施，鼓励他们推荐新客户。社交媒体和网络平台：通过社交媒体（如微信、微博等）进行宣传和互动，吸引潜在客户。线下活动：参加或组织社区活动、讲座、产说会等，直接与潜在客户接触。合作伙伴推荐：与财务规划师、会计师、律师等专业人士建立合作关系，通过他们推荐客户。3. 利用开门红的特点吸引客户产品优势宣传：强调开门红产品的高收益、稳定增长等特点。限时优惠：强调这些产品只在特定时间内提供，创造紧迫感。4. 提高服务质量和专业知识专业培训：确保自己对产品有深入的了解和专业的培训，能够解答客户的各种疑问。优质服务：通过提供优质的客户服务和专业建议，建立良好的口碑。5. 持续跟进和维护客户关系定期沟通：与潜在客户保持联系，了解他们的需求变化。客户关怀：定期发送行业信息、节日祝福等，维护与客户的良好关系。通过上述方法，您可以在开门红期间有效地找到并吸引目标客户。记住，每个客户的需求是独特的，因此，个性化的服务和专业的建议是关键。\n\n【标题】：客户说我有社保，不需要保险怎么办\n【内容】：例如，在公司提供的理赔年报中寻找与你的客户年龄、职业相仿的案例，以聊天的方式与客户进行探讨，询问他对案例的想法。&nbsp;",
        "requirement": ""
    }
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
                # print(f"========{model_name}_prediction finished============")
                # print(f"{predictions[0]}")
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
    template_file = "template/template.json"

    question = test_case["question"]
    requirement = test_case["requirement"]
    context = test_case["contexts"]
    debug_agent = BaseLiteLLMAgent(template_file="template/template_debug.json", model_invoke=get_qwen_response)
    rag_agent = BaseLiteLLMAgent(template_file=template_file, model_invoke=get_qwen_response)
    predictions = debug_agent.invoke(adapter_name=model_adapter_name_map[model_name],
                                        **{"question": test_case["question"]})
    print("**" * 20 + model_name + "**" * 20)
    print(predictions)
    print("=="*20)
    predictions = rag_agent.invoke(adapter_name=model_adapter_name_map[model_name],
                                        **{"question": question
                                            , "requirement": requirement
                                            , "context": context})
    print(predictions)

    print("**"*40 + "\n\n")

if __name__ == '__main__':

    test_single_agent("qwen")
    test_single_agent("qwen-rag-0527-exp2")
    test_single_agent("qwen-rag-0601-dpo-exp1")

    time_start = time.time()
    run_rag_comparison(
        data_dir="dataset/evaluation_dataset",
        output_dir="output/evaluation_dataset",
        template_file="template/template.json",
        model_names=["qwen","qwen-rag-0527-exp2","qwen-rag-0601-dpo-exp1"],
        max_samples=8,
        model_invoke=get_qwen_response
    )
    time_end = time.time()
    print(f"total time cost: {(time_end - time_start)}")


