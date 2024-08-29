# -*- encoding: utf-8 -*-
"""
@File    : test_inference.py
@Time    : 29/8/2024 16:43
@Author  : Duxy
@Email   : du.xi.yang@qq.com
@Software: PyCharm
"""
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

model = LLM(model="/root/autodl-tmp/qwen/Qwen1_5-14B-Chat-GPTQ-Int4", enable_lora=True)

lora_request = LoRARequest("stage2",
                           1,
                           "/root/autodl-tmp/checkpoints/qwen/0722_qwen15_rag_sft_exp1")


sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=256,
    stop=["[/assistant]"]
)

prompts = [
    "[user] Your prompt here [/user] [assistant]"
]
