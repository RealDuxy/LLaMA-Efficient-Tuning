# -*- encoding: utf-8 -*-
"""
@File    : preprocess.py
@Time    : 13/3/2024 14:11
@Author  : Duxy
@Email   : du.xi.yang@qq.com
@Software: PyCharm
"""
import json
import re


def convert_ranker_results_into_comparison_datasets(ranker_results, save_file):
    """
    将排序后的结果转换成对比数据集的格式。对比数据集的格式如下：
    [
        {
            "system": "同askbob_0222_6k",
            "instruction": "同askbob_0222_6k",
            "input": "",
            "output": ["xxx", "xxx"],(排序后的结果，由好到差)
            "history": []
        },
        ......
    ]

    需要注意：
    1. 【排序结果的问题】需要与【原始问题】对应
    2. 从instruction中提取【原始问题】可以使用extract_context_and_question_from_instruction方法。
    2. 如果A更好，则output_a排在前面，反之则output_b在前面
    3. 如果A和B一样好，则不要将这个数据放入对比数据集中
    4. 中间过程打印清楚

    :param ranker_results:
    :return:
    """

    def extract_context_and_question_from_instruction(text):
        # 使用正则表达式提取检索资料和用户问题
        search_info_pattern = r"```(.*?)```"
        question_pattern = r"解答用户的问题：(.*?)如果检索资料可以支持"

        # 使用非贪婪模式进行匹配，确保正确提取每个检索资料
        search_info_matches = re.findall(search_info_pattern, text, flags=re.DOTALL)
        # 提取用户问题
        question_match = re.search(question_pattern, text, flags=re.DOTALL)

        question = question_match.group(1).strip() if question_match else None

        return search_info_matches, question

    ...


def convert_ranker_results_into_judge_datasets(ranker_results, save_file):
    """
    将排序后的结果转换成审查排序数据集的格式。审查排序数据集的格式如下：
    [
        {
            "system": ranker_system,
            "instruction": ranker_prompt（字段填充后）,
            "input": "",
            "output": "GPT4模型的回复文本" (未解析)
            "history": []
        },
        ......
    ]

    需要注意：
    1. 无论是A好还是B好还是一样，全部问题都添加

    :param ranker_results:
    :return:
    """

    def extract_context_and_question_from_instruction(text):
        # 使用正则表达式提取检索资料和用户问题
        search_info_pattern = r"```(.*?)```"
        question_pattern = r"解答用户的问题：(.*?)如果检索资料可以支持"

        # 使用非贪婪模式进行匹配，确保正确提取每个检索资料
        search_info_matches = re.findall(search_info_pattern, text, flags=re.DOTALL)
        # 提取用户问题
        question_match = re.search(question_pattern, text, flags=re.DOTALL)

        question = question_match.group(1).strip() if question_match else None

        return search_info_matches, question

if __name__ == '__main__':
    data_file = "../askbob_0321_4k_comparison.json"

    datas = json.load(open(data_file, encoding="utf-8"))
    for i, data in enumerate(datas):
        datas[i]["output"] = datas[i].pop("outputs")
    json.dump(datas, open(data_file, "w", encoding="utf-8"), ensure_ascii=False, indent=4)


    # for i, data in enumerate(datas):
    #     output = json.loads(data["output"])
    #     output = output["choices"][0]["message"]["content"]
    #     datas[i]["output"] = output
    #
    # json.dump(datas, open("askbob_0321_4k_comparison_judge_response.json", "w", encoding="utf-8"), ensure_ascii=False, indent=4)






