# -*- encoding: utf-8 -*-
"""
@File    : train_sft.py
@Time    : 11/12/2023 15:09
@Author  : Duxy
@Email   : du.xi.yang@qq.com
@Software: PyCharm
"""
import math
from types import MethodType
from typing import Optional, Dict, Any

import torch
from transformers import AutoTokenizer, AutoConfig, PreTrainedTokenizerBase, BitsAndBytesConfig, AutoModelForCausalLM, \
    PreTrainedModel, PretrainedConfig
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.models.llama import modeling_llama as LlamaModule
from transformers.utils.versions import require_version

from utils import get_train_args, logger, get_current_device, prepare_model_for_training, init_adapter

from transformers.utils import (
    is_torch_bf16_cpu_available,
    is_torch_bf16_gpu_available,
    is_torch_cuda_available,
)

_is_fp16_available = is_torch_cuda_available()
_is_bf16_available = is_torch_bf16_gpu_available() or is_torch_bf16_cpu_available()


def infer_optim_dtype(model_dtype: torch.dtype) -> torch.dtype:
    r"""
    Infers the optimal dtype according to the model_dtype and device compatibility.
    """
    if _is_bf16_available and model_dtype == torch.bfloat16:
        return torch.bfloat16
    elif _is_fp16_available:
        return torch.float16
    else:
        return torch.float32

def train(args: Optional[Dict[str, Any]] = None):
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)

    # load tokenizer
    config_kwargs = {
        "trust_remote_code": True,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.hf_hub_token
    }

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
        split_special_tokens=model_args.split_special_tokens,
        padding_side="right", # training with left-padded tensors in fp16 precision may cause overflow
        **config_kwargs
    )

    # load model config
    if finetuning_args.finetuning_type != "lora" and model_args.checkpoint_dir is not None:
        logger.info("Use `model_name_or_path` to specify the model trained with full/freeze method.")
        model_to_load = model_args.checkpoint_dir[0]
    else:
        model_to_load = model_args.model_name_or_path

    config = AutoConfig.from_pretrained(model_to_load, **config_kwargs)
    # Fix tokenizer (for ChatGLM2 and ChatGLM3)
    if getattr(config, "model_type", None) == "chatglm":
        tokenizer._pad = MethodType(PreTrainedTokenizerBase._pad, tokenizer)

    '''=================================初始化模型的一些设置==================================='''
    model_args.compute_dtype = infer_optim_dtype(getattr(config, "torch_dtype", None))
    setattr(config, "torch_dtype", model_args.compute_dtype)

    '''=============RoPE 设置============='''
    # rope_scaling: linear scaling < NTK-aware scaling < dynamic NTK-aware scaling
    # linear scaling：位置索引直接被scaling factor切分，需要微调才能有好效果
    # NTK-aware scaling: 神经切线核：linear scaling的改进，微调和不微调都会有不错的效果
    # dynamic NTK-aware scaling：参数固定时，短文本不会降低效果，长文本效果好；参数不固定时不确定

    # 当input_length小于模型限制，默认scaling factor=1.0
    if model_args.rope_scaling is not None:
        if not hasattr(config, "rope_scaling"):
            logger.warning("Current model does not support RoPE scaling.")
        else:
            if training_args.do_train:
                if model_args.rope_scaling == "dynamic":
                    logger.warning(
                        "Dynamic NTK may not work well with fine-tuning. "
                        "See: https://github.com/huggingface/transformers/pull/24653"
                    )

                current_max_length = getattr(config, "max_position_embeddings", None)
                if current_max_length and model_args.model_max_length > current_max_length:
                    scaling_factor = float(math.ceil(model_args.model_max_length / current_max_length))
                else:
                    logger.warning("Input length is smaller than max length. Consider increase input length.")
                    scaling_factor = 1.0
            else:
                scaling_factor = 2.0

            setattr(config, "rope_scaling", {"type": model_args.rope_scaling, "factor": scaling_factor})
            logger.info("Using {} scaling strategy and setting scaling factor to {}".format(
                model_args.rope_scaling, scaling_factor
            ))

    # if model_args.flash_attn:
    #     if getattr(config, "model_type", None) == "llama":
    #         if is_flash_attn2_available():
    #             LlamaModule.LlamaAttention = LlamaPatches.LlamaFlashAttention2
    #             LlamaModule.LlamaModel._prepare_decoder_attention_mask = LlamaPatches._prepare_decoder_attention_mask
    #             logger.info("Using FlashAttention-2 for faster training and inference.")
    #         else:
    #             logger.warning("FlashAttention-2 is not installed.")
    #     elif getattr(config, "model_type", None) in ["qwen", "Yi"]:
    #         logger.info("Current model automatically enables FlashAttention if installed.")
    #     else:
    #         logger.warning("Current model does not support FlashAttention.")
    # elif is_trainable and model_args.shift_attn and getattr(config, "model_type", None) == "llama":
    #     LlamaModule.LlamaAttention = LlamaPatches.LlamaShiftShortAttention
    #     logger.warning("Using `--flash_attn` for faster training in large context length.")

    # Set shift short attention (S^2-Attn)
    if training_args.do_train and model_args.shift_attn:
        if getattr(config, "model_type", None) == "llama":
            setattr(config, "group_size_ratio", 0.25)
            logger.info("Using shift short attention with group_size_ratio=1/4.")
        else:
            logger.warning("Current model does not support shift short attention.")

    # Quantization configurations (using bitsandbytes library)
    if model_args.quantization_bit is not None:
        if getattr(config, "quantization_config", None):
            raise ValueError("Remove `quantization_bit` if you are using a quantized model.")

        if is_deepspeed_zero3_enabled():
            raise ValueError("DeepSpeed ZeRO-3 is incompatible with quantization.")

        if model_args.quantization_bit == 8:
            require_version("bitsandbytes>=0.37.0", "To fix: pip install bitsandbytes>=0.37.0")
            config_kwargs["load_in_8bit"] = True
            config_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        if model_args.quantization_bit == 4:
            require_version("bitsandbytes>=0.39.0", "To fix: pip install bitsandbytes>=0.39.0")
            config_kwargs["load_in_4bit"] = True
            config_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=model_args.compute_dtype,
                bnb_4bit_use_double_quant=model_args.double_quantization,
                bnb_4bit_quant_type=model_args.quantization_type
            )

        config_kwargs["device_map"] = {"": get_current_device()}
        logger.info("Quantizing model to {} bit.".format(model_args.quantization_bit))

    # Load pre-trained models (without valuehead)
    model = AutoModelForCausalLM.from_pretrained(
        model_to_load,
        config=config,
        torch_dtype=model_args.compute_dtype,
        low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),
        **config_kwargs
    )

    # Disable custom generate method (for Qwen and Baichuan2)
    if isinstance(model, PreTrainedModel) and "GenerationMixin" not in str(model.generate.__func__):
        model.generate = MethodType(PreTrainedModel.generate, model)

    # Fix LM head (for ChatGLM2 and ChatGLM3)
    if getattr(config, "model_type", None) == "chatglm":
        setattr(model, "lm_head", model.transformer.output_layer)
        setattr(model, "_keys_to_ignore_on_save", ["lm_head.weight"])

    # Register auto class to save the custom code files
    if isinstance(config, PretrainedConfig) and "AutoConfig" in getattr(config, "auto_map", {}):
        config.__class__.register_for_auto_class()
    if isinstance(model, PreTrainedModel) and "AutoModelForCausalLM" in getattr(config, "auto_map", {}):
        model.__class__.register_for_auto_class()
    if isinstance(tokenizer, PreTrainedTokenizerBase) and "AutoTokenizer" in tokenizer.init_kwargs.get("auto_map", {}):
        tokenizer.__class__.register_for_auto_class()


    # Initialize adapters
    model = prepare_model_for_training(model=model, finetuning_args=finetuning_args) if training_args.do_train else model
    model = init_adapter(model, model_args, finetuning_args, training_args.do_train)
    model = model.train() if is_trainable else model.eval()


