# -*- encoding: utf-8 -*-
"""
@File    : test.py
@Time    : 16/5/2024 20:59
@Author  : Duxy
@Email   : du.xi.yang@qq.com
@Software: PyCharm
"""

from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForVision2Seq, AutoProcessor, AutoTokenizer

model_path = "/mnt/e/UbuntuFiles/models_saved/Qwen1.5-14B-Chat-GPTQ-Int4"
model = AutoModelForCausalLM.from_pretrained(
        model_path,trust_remote_code=True
    )

