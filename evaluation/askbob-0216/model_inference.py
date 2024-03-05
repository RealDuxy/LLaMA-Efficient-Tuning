# -*- encoding: utf-8 -*-
"""
@File    : model_inference.py
@Time    : 5/3/2024 13:31
@Author  : Duxy
@Email   : du.xi.yang@qq.com
@Software: PyCharm
"""

"""
This script implements an API for the ChatGLM3-6B model,
formatted similarly to OpenAI's API (https://platform.openai.com/docs/api-reference/chat).
It's designed to be run as a web server using FastAPI and uvicorn, making the ChatGLM3-6B model accessible through HTTP requests.

Key Components and Features:
- Model and Tokenizer Setup: Configures the model and tokenizer paths and loads them, utilizing GPU acceleration if available.
- FastAPI Configuration: Sets up a FastAPI application with CORS middleware for handling cross-origin requests.
- API Endpoints:
  - "/v1/models": Lists the available models, specifically ChatGLM3-6B.
  - "/v1/chat/completions": Processes chat completion requests with options for streaming and regular responses.
- Token Limit Caution: In the OpenAI API, 'max_tokens' is equivalent to HuggingFace's 'max_new_tokens', not 'max_length'.
For instance, setting 'max_tokens' to 8192 for a 6b model would result in an error due to the model's inability to output
that many tokens after accounting for the history and prompt tokens.
- Stream Handling and Custom Functions: Manages streaming responses and custom function calls within chat responses.
- Pydantic Models: Defines structured models for requests and responses, enhancing API documentation and type safety.
- Main Execution: Initializes the model and tokenizer, and starts the FastAPI app on the designated host and port.

Note: This script doesn't include the setup for special tokens or multi-GPU support by default.
Users need to configure their special tokens and can enable multi-GPU support as per the provided instructions.

"""

import os
import time
from contextlib import asynccontextmanager
from typing import List, Literal, Optional, Union

import pandas as pd
from peft import PeftModel
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
# from loguru import logger
from pydantic import BaseModel, Field
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# from utils import process_response, generate_chatglm3, generate_stream_chatglm3
from API_reference import generate_answer_from_glm4

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from sse_starlette.sse import EventSourceResponse

# Set up limit request time
EventSourceResponse.DEFAULT_PING_INTERVAL = 1000

# MODEL_PATH = os.environ.get('MODEL_PATH', 'chatglm3')
# TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", MODEL_PATH)
# PEFT_MODEL_PATH = os.environ.get('MODEL_PATH', 'chatglm3_lora')
# MODEL_PATH = "/root/.cache/modelscope/hub/ZhipuAI/chatglm3-6b"
# TOKENIZER_PATH = "/root/.cache/modelscope/hub/ZhipuAI/chatglm3-6b"
# PEFT_MODEL_PATH = "model_sft/ins_moderation/checkpoint-100"

type_t='''原文：5 
 相似：5  
抽取：10
 综合推理：10'''

from modeling_chatglm import ChatGLMForConditionalGeneration


def prepare_model_for_half_training(model, output_embedding_layer_name="lm_head",
                                    use_gradient_checkpointing=True, layer_norm_names=["layer_norm"]):
    #  不要使用 model.half(), 这样会先截取精度再训练了, 最初data就要保持half
    for name, param in model.named_parameters():
        # freeze base model's layers
        param.requires_grad = False
        # cast layer norm in fp32 for stability for 8bit models
        if param.ndim == 1 and any(layer_norm_name in name for layer_norm_name in layer_norm_names):
            param.data = param.data.to(torch.float32)
        elif output_embedding_layer_name in name:  # lm_head也需要是tf.float32(最后一层)
            param.data = param.data.to(torch.float32)
        else:
            param.data = param.data.to(torch.half)

    if use_gradient_checkpointing:
        # For backward compatibility
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        # enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()
    return model


def run_chatglm_predict_askbob0126(model_path, tokenizer_path, post_fix, peft_path=None, data_path="askbob-0216.xlsx"):

    # 加载模型
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)  # , trust_remote_code=True
    # model = AutoModel.from_pretrained(MODEL_PATH, device_map="auto", trust_remote_code=True).eval()#, trust_remote_code=True
    model = ChatGLMForConditionalGeneration.from_pretrained(model_path, device_map="auto").eval()

    model = prepare_model_for_half_training(model,
                                            use_gradient_checkpointing=False,
                                            output_embedding_layer_name="lm_head",  # output_layer
                                            layer_norm_names=["post_attention_layernorm",
                                                              "final_layernorm",
                                                              "input_layernorm",
                                                              ])

    # model.gradient_checkpointing_enable()
    # model.enable_input_require_grads()
    model.is_parallelizable = True
    model.model_parallel = True
    model.config.use_cache = True
    if peft_path:
        model = PeftModel.from_pretrained(model, peft_path)
    model = model.cuda(0)

    df = pd.read_excel(data_path).to_dict("records")

    new_df = []
    for line in tqdm(df):
        question = line["评估问题"]
        if isinstance(question, str):
            history_prompt = eval(line["prompt"])
            history = history_prompt["history"]
            prompt = history_prompt["prompt"]
            seg_prompt = prompt.split("\n\n")
            seg_prompt[-1] = question
            prompt = "\n\n".join(seg_prompt)
            # output = line["评估输出"]
            output, history = model.chat(tokenizer, prompt, history=history, temperature=0.1)
            # output = generate_answer_from_glm4(history, prompt)
            # line["评估输出"] = output.strip()


            first_round_retrieval_results = line["final_retrieval_result"]
            first_round_retrieval_results = eval(first_round_retrieval_results)[:-1]
            context = ""
            for x in first_round_retrieval_results:
                context += x + "\n\n"
            extract_type = line[type_t]
            # output = line["评估输出"]
            new_df.append({
                "问题": question,
                "引文": context,
                "问题类型": extract_type,
                "回答": output
            })
    save_path = data_path.replace(".", f"-{post_fix}.")
    pd.DataFrame(new_df).to_excel(save_path)

if __name__ == '__main__':
    run_chatglm_predict_askbob0126(
        model_path="/root/autodl-tmp/chatglm3-6b",
        tokenizer_path="/root/autodl-tmp/chatglm3-6b",
        post_fix="0304_vanilla_dpo",
        peft_path="../checkpoints/0304_vanilla_dpo"
    )

    run_chatglm_predict_askbob0126(
        model_path="/root/autodl-tmp/chatglm3-6b",
        tokenizer_path="/root/autodl-tmp/chatglm3-6b",
        post_fix="0304_vanilla_dpo",
        peft_path="../checkpoints/0304_vanilla_dpo"
    )