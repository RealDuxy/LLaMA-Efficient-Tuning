# -*- encoding: utf-8 -*-
"""
@File    : model_inference.py
@Time    : 5/3/2024 13:31
@Author  : Duxy
@Email   : du.xi.yang@qq.com
@Software: PyCharm
"""
import argparse
import json
import random

import transformers

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
from contextlib import asynccontextmanager, nullcontext
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
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
# from utils import process_response, generate_chatglm3, generate_stream_chatglm3


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


def run_chatglm_predict_askbob0126(model_path, tokenizer_path, post_fix, peft_path=None, data_path="askbob-0126.xlsx"):

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)  # , trust_remote_code=True
    # model = AutoModel.from_pretrained(MODEL_PATH, device_map="auto", trust_remote_code=True).eval()#, trust_remote_code=True
    model = AutoModelForCausalLM.from_pretrained(
        model_path,trust_remote_code=True
    )

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

def run_qwen_predict_askbob0126(model_path, tokenizer_path, post_fix, peft_path=None, data_path="askbob-0126.xlsx"):
    device = "cuda:0"
    # offload_folder = "./offload"
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto",torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # model = prepare_model_for_half_training(model,
    #                                         use_gradient_checkpointing=False,
    #                                         output_embedding_layer_name="lm_head",  # output_layer
    #                                         layer_norm_names=["post_attention_layernorm",
    #                                                           "final_layernorm",
    #                                                           "input_layernorm",
    #                                                           ])
    LAYERNORM_NAMES = ["norm", "ln"]
    # model = prepare_model_for_half_training(model,
    #                                         use_gradient_checkpointing=False,
    #                                         output_embedding_layer_name="lm_head",  # output_layer
    #                                         layer_norm_names=LAYERNORM_NAMES)

    # model.gradient_checkpointing_enable()
    # model.enable_input_require_grads()
    # model.is_parallelizable = True
    # model.model_parallel = True
    # model.config.use_cache = True
    if peft_path:
        model = PeftModel.from_pretrained(model, model_id=peft_path)

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

            messages = history + [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = tokenizer([text], return_tensors="pt").to("cuda")
            generated_ids = model.generate(model_inputs.input_ids,
                                           max_new_tokens=1024,
                                           do_sample=True,
                                           pad_token_id=tokenizer.eos_token_id)

            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in
                             zip(model_inputs.input_ids, generated_ids)]

            output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


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

def run_chatglm_predict_askbobqa_3_times(model_path,
                                         tokenizer_path,
                                         post_fix,
                                         peft_path=None,
                                         data_path="../../data/askbob_qa/askbob_0222_6k.json",
                                         **kwargs,
                                         ):
    def load_all(device=0):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)  # , trust_remote_code=True
        # model = AutoModel.from_pretrained(MODEL_PATH, device_map="auto", trust_remote_code=True).eval()#, trust_remote_code=True
        model = AutoModelForCausalLM.from_pretrained(
            model_path,trust_remote_code=True
        )
        model = prepare_model_for_half_training(model,
                                                use_gradient_checkpointing=False,
                                                output_embedding_layer_name="lm_head",  # output_layer
                                                layer_norm_names=["post_attention_layernorm",
                                                                  "final_layernorm",
                                                                  "input_layernorm",
                                                                  ])

        # model.gradient_checkpointing_enable()
        # model.enable_input_require_grads()
        # model.is_parallelizable = True
        # model.model_parallel = True
        # model.config.use_cache = True
        if peft_path:
            model = PeftModel.from_pretrained(model, model_id=peft_path, trust_remote_code=True)
        model = model.to(f"cuda:{device}")

        return model,tokenizer

    s, e = kwargs["start"], kwargs["end"]
    device = kwargs["device"]

    print(f"runnning [{s,e}] of {data_path.split('/')[-1]}")
    model, tokenizer = load_all(device=0)

    with open(data_path, "r") as f:
        df = json.load(f)

    new_df = []
    for line in tqdm(df[s:e]):
        system = line["system"]
        if isinstance(system, str):
            history =  [{"role": "system", "content": system}]
            prompt = line["instruction"]
            outputs = []
            for _ in range(4):
                seed = random.randint(0, 10000)
                transformers.set_seed(seed)
                output, history = model.chat(tokenizer, prompt, history=history, temperature=0.9)
                outputs.append(output)
            # output = line["评估输出"]
            new_df.append(line.update({"output":outputs}))
    save_path = data_path.replace(".", f"-{post_fix}-{s,e}.")
    pd.DataFrame(new_df).to_excel(save_path)


if __name__ == '__main__':
    # model_path = "/mnt/d/PycharmProjects/models/chatglm3-6b"
    # run_chatglm_predict_askbob0126(
    #     model_path=model_path,
    #     tokenizer_path=model_path,
    #     post_fix="0304_vanilla_dpo",
    #     peft_path="../../checkpoints/0304_vanilla_dpo"
    # )
    #
    # run_chatglm_predict_askbob0126(
    #     model_path=model_path,
    #     tokenizer_path=model_path,
    #     post_fix="0304_chatglm3-stage1-ckpt282_dpo",
    #     peft_path="../../checkpoints/0304_chatglm3-stage1-ckpt282_dpo"
    # )
    # model_path = "/mnt/d/PycharmProjects/models/Qwen1.5-0.5B-Chat"
    # run_qwen_predict_askbob0126(
    #     model_path=model_path,
    #     tokenizer_path=model_path,
    #     post_fix="0304_qwen0-5B_stage1_spec_ft",
    #     peft_path="../../checkpoints/0304_qwen0-5B_stage1_spec_ft"
    # )

    # model_path = "/mnt/d/PycharmProjects/models/Qwen1.5-7B-Chat"
    # run_qwen_predict_askbob0126(
    #     model_path=model_path,
    #     tokenizer_path=model_path,
    #     post_fix="0304_qwen7B_stage1_spec_ft",
    #     peft_path="../../checkpoints/0304_qwen7B_stage1_spec_ft"
    # )

    parser = argparse.ArgumentParser(description='Evaluation')

    parser.add_argument('--device', type=int, default=0,
                        help='device'
                             "0,1,2,3")

    args = parser.parse_args()

    model_path = "/mnt/e/UbuntuFiles/models_saved/chatglm3/"
    data_path = "../../data/askbob_qa/askbob_0222_6k.json"
    data_nums = len(json.load(open(data_path, "r")))
    device = args.device

    print("="*20)
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
    os.environ["CUDA_VISIBLE_DEVICES"]= str(device)
    print(f"setting CUDA_VISIBLE_DEVICES to {os.environ['CUDA_VISIBLE_DEVICES']}")
    print("=" * 20)

    range_options = list(range(0, data_nums, data_nums//4))
    s, e = range_options[device], range_options[device]+data_nums//4

    run_chatglm_predict_askbobqa_3_times(
        model_path=model_path,
        tokenizer_path=model_path,
        post_fix="stage1_askbobqa_3_times",
        peft_path="../../checkpoints/0205_stage1_spec_ft",
        data_path=data_path,
        device=device, start=s, end=e,

    )