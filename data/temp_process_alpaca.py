# -*- encoding: utf-8 -*-
"""
@File    : temp_process_alpaca.py
@Time    : 13/3/2024 19:57
@Author  : Duxy
@Email   : du.xi.yang@qq.com
@Software: PyCharm
"""
import json


def main(data_path):

    lines = json.load(open(data_path, 'r', encoding='utf-8'))

    lines = [line.update({"history":[]}) for line in lines]

    with open(data_path, 'w', encoding='utf-8') as wf:
        json.dump(lines, wf, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    main("al")
