# -*- coding: utf-8 -*-
"""
Created on 2024/6/5 14:40
author: ruanzhihao_archfool
"""

import os
from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda" # the device to load the model onto


root_dir = '/mnt/d/PycharmProjects/LLaMA-Efficient-Tuning'
data_dir = os.path.join(root_dir, 'data')
llm_data_dir = "/mnt/d/PycharmProjects/models/"

QWEN1PLUS5_14B_INT4 = 'Qwen1.5-14B-Chat-GPTQ-Int4'
qwen_model_dir = os.path.join(llm_data_dir, QWEN1PLUS5_14B_INT4)

if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained(
        qwen_model_dir,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(qwen_model_dir)

    prompt = "Give me a short introduction to large language model."
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    print(tokenizer.convert_ids_to_tokens(model_inputs.input_ids.cpu().numpy()[0]))

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
