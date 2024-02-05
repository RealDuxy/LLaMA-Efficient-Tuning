# -*- encoding: utf-8 -*-
"""
@File    : utils.py
@Time    : 11/12/2023 15:18
@Author  : Duxy
@Email   : du.xi.yang@qq.com
@Software: PyCharm
"""
import logging
import os
import sys
from typing import Optional, Dict, Any, Tuple

import datasets
import torch
import transformers
from transformers import HfArgumentParser, Seq2SeqTrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from llmtuner.hparams import ModelArguments, DataArguments, FinetuningArguments, GeneratingArguments

from peft import PeftModel

_TRAIN_CLS = Tuple[
    ModelArguments, DataArguments, Seq2SeqTrainingArguments, FinetuningArguments, GeneratingArguments
]
_TRAIN_ARGS = [
    ModelArguments, DataArguments, Seq2SeqTrainingArguments, FinetuningArguments, GeneratingArguments
]


def set_logger():
    logger_name = __name__
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S"
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    return logger


logger = set_logger()


def parse_args(parser: "HfArgumentParser", args: Optional[Dict[str, Any]] = None) -> Tuple[Any]:
    if args is not None:
        return parser.parse_dict(args)
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        return parser.parse_yaml_file(os.path.abspath(sys.argv[1]))
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        return parser.parse_json_file(os.path.abspath(sys.argv[1]))
    else:
        return parser.parse_args_into_dataclasses()


def parse_train_args(args: Optional[Dict[str, Any]] = None) -> _TRAIN_CLS:
    parser = HfArgumentParser(_TRAIN_ARGS)
    return parse_args(parser, args)


def get_current_device() -> str:
    import accelerate
    dummy_accelerator = accelerate.Accelerator()
    if accelerate.utils.is_xpu_available():
        return "xpu:{}".format(dummy_accelerator.local_process_index)
    else:
        return dummy_accelerator.local_process_index if torch.cuda.is_available() else "cpu"


def _verify_model_args(model_args: "ModelArguments", finetuning_args: "FinetuningArguments") -> None:
    if model_args.quantization_bit is not None and finetuning_args.finetuning_type != "lora":
        raise ValueError("Quantization is only compatible with the LoRA method.")

    if (
            model_args.checkpoint_dir is not None
            and len(model_args.checkpoint_dir) != 1
            and finetuning_args.finetuning_type != "lora"
    ):
        raise ValueError("Multiple checkpoints are only available for LoRA tuning.")


def get_train_args(args: Optional[Dict[str, Any]] = None) -> _TRAIN_CLS:
    model_args, data_args, training_args, finetuning_args, generating_args = parse_train_args(args)

    # Setup logging
    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Check arguments
    data_args.init_for_training(training_args.seed)

    if finetuning_args.stage != "pt" and data_args.template is None:
        raise ValueError("Please specify which `template` to use.")

    if finetuning_args.stage != "sft" and training_args.predict_with_generate:
        raise ValueError("`predict_with_generate` cannot be set as True except SFT.")

    if finetuning_args.stage == "sft" and training_args.do_predict and not training_args.predict_with_generate:
        raise ValueError("Please enable `predict_with_generate` to save model predictions.")

    if finetuning_args.stage in ["rm", "ppo"] and training_args.load_best_model_at_end:
        raise ValueError("RM and PPO stages do not support `load_best_model_at_end`.")

    if finetuning_args.stage == "ppo" and not training_args.do_train:
        raise ValueError("PPO training does not support evaluation, use the SFT stage to evaluate models.")

    if finetuning_args.stage in ["rm", "dpo"] and (
            not all([data_attr.ranking for data_attr in data_args.dataset_list])):
        raise ValueError("Please use ranked datasets for reward modeling or DPO training.")

    if finetuning_args.stage == "ppo" and model_args.shift_attn:
        raise ValueError("PPO training is incompatible with S^2-Attn.")

    if training_args.max_steps == -1 and data_args.streaming:
        raise ValueError("Please specify `max_steps` in streaming mode.")

    if training_args.do_train and training_args.predict_with_generate:
        raise ValueError("`predict_with_generate` cannot be set as True while training.")

    if training_args.do_train and finetuning_args.finetuning_type == "lora" and finetuning_args.lora_target is None:
        raise ValueError("Please specify `lora_target` in LoRA training.")

    _verify_model_args(model_args, finetuning_args)

    if training_args.do_train and model_args.quantization_bit is not None and (not finetuning_args.upcast_layernorm):
        logger.warning("We recommend enable `upcast_layernorm` in quantized training.")

    if training_args.do_train and (not training_args.fp16) and (not training_args.bf16):
        logger.warning("We recommend enable mixed precision training.")

    if (not training_args.do_train) and model_args.quantization_bit is not None:
        logger.warning("Evaluating model in 4/8-bit mode may cause lower scores.")

    if (not training_args.do_train) and finetuning_args.stage == "dpo" and finetuning_args.ref_model is None:
        logger.warning("Specify `ref_model` for computing rewards at evaluation.")

    # postprocess training_args
    if (
            training_args.local_rank != -1
            and training_args.ddp_find_unused_parameters is None
            and finetuning_args.finetuning_type == "lora"
    ):
        logger.warning("`ddp_find_unused_parameters` needs to be set as False for LoRA in DDP training.")
        training_args_dict = training_args.to_dict()
        training_args_dict.update(dict(ddp_find_unused_parameters=False))
        training_args = Seq2SeqTrainingArguments(**training_args_dict)

    if (
            training_args.resume_from_checkpoint is None
            and training_args.do_train
            and os.path.isdir(training_args.output_dir)
            and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError("Output directory already exists and is not empty. Please set `overwrite_output_dir`.")

        if last_checkpoint is not None:
            training_args_dict = training_args.to_dict()
            training_args_dict.update(dict(resume_from_checkpoint=last_checkpoint))
            training_args = Seq2SeqTrainingArguments(**training_args_dict)
            logger.info("Resuming training from {}. Change `output_dir` or use `overwrite_output_dir` to avoid.".format(
                training_args.resume_from_checkpoint
            ))

    if finetuning_args.stage in ["rm", "ppo"] and training_args.resume_from_checkpoint is not None:
        logger.warning("Add {} to `checkpoint_dir` to resume training from checkpoint.".format(
            training_args.resume_from_checkpoint
        ))

    # postprocess model_args
    model_args.compute_dtype = (
        torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else None)
    )
    model_args.model_max_length = data_args.cutoff_len

    # Log on each process the small summary:
    logger.info("Process rank: {}, device: {}, n_gpu: {}\n  distributed training: {}, compute dtype: {}".format(
        training_args.local_rank, training_args.device, training_args.n_gpu,
        bool(training_args.local_rank != -1), str(model_args.compute_dtype)
    ))
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    transformers.set_seed(training_args.seed)

    return model_args, data_args, training_args, finetuning_args, generating_args


def load_model():
    ...


def init_adapter(
        model: "PreTrainedModel",
        model_args: "ModelArguments",
        finetuning_args: "FinetuningArguments",
        is_trainable: bool
) -> "PreTrainedModel":
    r"""
    Initializes the adapters.

    Support full-parameter and LoRA training.

    Note that the trainable parameters must be cast to float32.
    """

    if (not is_trainable) and model_args.checkpoint_dir is None:
        logger.info("Checkpoint is not found at evaluation, load the original model.")
        return model

    if finetuning_args.finetuning_type == "full" and is_trainable:
        logger.info("Fine-tuning method: Full")
        model = model.float()

    if finetuning_args.finetuning_type == "lora":
        logger.info("Fine-tuning method: LoRA")
        checkpoint_to_resume = None

        if model_args.checkpoint_dir is not None:
            is_mergeable = True
            if getattr(model, "quantization_method", None) == "gptq":
                assert len(model_args.checkpoint_dir) == 1, "GPTQ quantized model only accepts a single checkpoint."
                is_mergeable = False

            if (is_trainable and finetuning_args.resume_lora_training) or (not is_mergeable):
                checkpoints_to_merge, checkpoint_to_resume = model_args.checkpoint_dir[:-1], model_args.checkpoint_dir[
                    -1]
            else:
                checkpoints_to_merge = model_args.checkpoint_dir

            for checkpoint in checkpoints_to_merge:
                model = PeftModel.from_pretrained(model, checkpoint)
                model = model.merge_and_unload()

            if len(checkpoints_to_merge) > 0:
                logger.info("Merged {} model checkpoint(s).".format(len(checkpoints_to_merge)))

            if checkpoint_to_resume is not None:  # resume lora training
                model = PeftModel.from_pretrained(model, checkpoint_to_resume, is_trainable=is_trainable)

        if is_trainable and checkpoint_to_resume is None:  # create new lora weights while training
            if len(finetuning_args.lora_target) == 1 and finetuning_args.lora_target[0] == "all":
                target_modules = find_all_linear_modules(model, model_args.quantization_bit)
            else:
                target_modules = finetuning_args.lora_target

            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=finetuning_args.lora_rank,
                lora_alpha=finetuning_args.lora_alpha,
                lora_dropout=finetuning_args.lora_dropout,
                target_modules=target_modules,
                modules_to_save=finetuning_args.additional_target
            )
            model = get_peft_model(model, lora_config)

    if model_args.checkpoint_dir is not None:
        logger.info("Loaded fine-tuned model from checkpoint(s): {}".format(",".join(model_args.checkpoint_dir)))

    return model


def prepare_model_for_training(
        model: "PreTrainedModel",
        finetuning_args: "FinetuningArguments",
        output_layer_name: Optional[str] = "lm_head",
        use_gradient_checkpointing: Optional[bool] = True,
) -> "PreTrainedModel":
    r"""
    Includes:
        (1) cast the layernorm in fp32
        (2) make output embedding layer require grads
        (3) upcast the lm_head to fp32
    Inspired by: https://github.com/huggingface/peft/blob/v0.2.0/src/peft/utils/other.py#L33
    """

    # upcast layernorm 为了训练稳定
    # LN中的方差和均方根操作可能会导致数值不稳定
    # 计算方差和平方根的时候，fp16动态范围较小，rounding误差会变大
    # LN的精度非常重要，对性能有显著影响，需要保证精度
    if finetuning_args.upcast_layernorm:
        for name, param in model.named_parameters():
            if param.ndim == 1 and any(ln_name in name for ln_name in {"norm", "ln"}):
                param.data = param.data.to(torch.float32)
        logger.info("Upcasting weights in layernorm in float32.")

    # 添加noise embedding，可以有效降低过拟合。
    if finetuning_args.neft_alpha > 1e-6:
        def neftune_forward_hook(module: torch.nn.Module, args: Tuple[torch.Tensor], output: torch.Tensor):
            if module.training:
                dims = torch.tensor(output.size(1) * output.size(2))
                mag_norm = finetuning_args.neft_alpha / torch.sqrt(dims)
                output = output + torch.zeros_like(output).uniform_(-mag_norm, mag_norm)
            return output

        model.get_input_embeddings().register_forward_hook(neftune_forward_hook)
        logger.info("Using noisy embedding with alpha={:.2f}".format(finetuning_args.neft_alpha))

    # output embedding layer直接决定输出，需要全量更新，提升模型表达能力
    if use_gradient_checkpointing and getattr(model, "supports_gradient_checkpointing", False):
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module: torch.nn.Module, args: Tuple[torch.Tensor], output: torch.Tensor):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        # 梯度检查点保存，可以减少内存消耗
        # 注意这里没有指定检查点，由模型内部决定
        model.gradient_checkpointing_enable()
        model.config.use_cache = False  # turn off when gradient checkpointing is enabled
        logger.info("Gradient checkpointing enabled.")

    if finetuning_args.finetuning_type != "full" and hasattr(model, output_layer_name):
        '''
        output layer权重类型转换
        输出的数据类型转换
        '''
        output_layer = getattr(model, output_layer_name)
        if isinstance(output_layer, torch.nn.Linear):
            def fp32_forward_pre_hook(module: torch.nn.Module, args: Tuple[torch.Tensor]):
                return args[0].to(output_layer.weight.dtype)

            def fp32_forward_post_hook(module: torch.nn.Module, args: Tuple[torch.Tensor], output: torch.Tensor):
                return output.to(torch.float32)

            output_layer.register_forward_pre_hook(fp32_forward_pre_hook)
            output_layer.register_forward_hook(fp32_forward_post_hook)

    return model
