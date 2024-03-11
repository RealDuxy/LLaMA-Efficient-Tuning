from contextlib import nullcontext

import requests
import torch
from peft import PeftModel
from pydantic import BaseModel, Field
from tenacity import wait_random_exponential, retry, stop_after_attempt
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "/mnt/d/PycharmProjects/models/chatglm3-6b/"
peft_path = "../../checkpoints/stepback_query_generation"


def load_models(model_path, peft_path):
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

        context_maybe_zero3 = nullcontext()
        with context_maybe_zero3:
            current_embedding_size = model.get_input_embeddings().weight.size(0)

        if len(tokenizer) > current_embedding_size:
            new_embedding_size = model.get_input_embeddings().weight.size(0)
            num_new_tokens = new_embedding_size - current_embedding_size
            print("Resized token embeddings from {} to {}.".format(current_embedding_size, new_embedding_size))
            model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)
            # with context_maybe_zero3:
            #     new_embedding_size = model.get_input_embeddings().weight.size(0)
            #     num_new_tokens = new_embedding_size - current_embedding_size
            #     _noisy_mean_initialization(model.get_input_embeddings().weight.data, num_new_tokens)
            #     _noisy_mean_initialization(model.get_output_embeddings().weight.data, num_new_tokens)

        # model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)

        return model

    # 加载模型
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)  # , trust_remote_code=True
    # model = AutoModel.from_pretrained(MODEL_PATH, device_map="auto", trust_remote_code=True).eval()#, trust_remote_code=True
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True
    )

    model = prepare_model_for_half_training(model,
                                            use_gradient_checkpointing=False,
                                            output_embedding_layer_name="lm_head",  # output_layer
                                            layer_norm_names=["post_attention_layernorm",
                                                              "final_layernorm",
                                                              "input_layernorm",
                                                              ])
    model.is_parallelizable = True
    model.model_parallel = True
    model.config.use_cache = True
    if peft_path:
        model = PeftModel.from_pretrained(model, peft_path)
    model = model.cuda(0)
    return model, tokenizer


model, tokenizer = load_models(model_path, peft_path)


@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(10))
def get_chatglm_response(messages):
    prompt = messages[-1]["content"]
    history = messages[:-1]
    output, history = model.chat(tokenizer, prompt, history=history, temperature=0.1)
    return output
