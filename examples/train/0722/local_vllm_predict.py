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

from vllm_wrapper import vLLMWrapper

model_path = "/root/autodl-tmp/qwen/Qwen1_5-14B-Chat-GPTQ-Int4"
vllm_model = vLLMWrapper(model_path,
                         dtype="float16",
                         tensor_parallel_size=1,
                         gpu_memory_utilization=0.9)

history=None
while True:
    Q=input('提问:')
    response, history = vllm_model.chat(query=Q,
                                        history=history)
    print(response)
    history=history[:20]
