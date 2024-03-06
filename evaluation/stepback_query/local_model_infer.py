import openai
import requests
import torch
from peft import PeftModel
from pydantic import BaseModel, Field
from tenacity import wait_random_exponential, retry, stop_after_attempt
from transformers import AutoTokenizer, AutoModelForCausalLM
from zhipuai import ZhipuAI

model_path = "/mnd/d/PycharmProjects/models/chatglm-6b/"
peft_path = None


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
