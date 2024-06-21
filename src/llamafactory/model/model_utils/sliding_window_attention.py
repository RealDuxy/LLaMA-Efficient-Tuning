# Copyright 2024 EleutherAI, HuggingFace Inc., Yukang Chen, and the LlamaFactory team.
#
# This code is based on the EleutherAI's GPT-NeoX and the HuggingFace's Transformers libraries.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/llama/modeling_llama.py
# This code is also inspired by the original LongLoRA implementation.
# https://github.com/dvlab-research/LongLoRA/blob/main/llama_attn_replace.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import inspect
import warnings

import math
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn
from transformers.models.qwen2.modeling_qwen2 import (
    Cache,
    Qwen2FlashAttention2,
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.utils import logging
from transformers.utils.versions import require_version

from ...extras.constants import SUPPORTED_CLASS_FOR_S2ATTN, SUPPORTED_CLASS_FOR_SWATTN
from ...extras.logging import get_logger

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from ...hparams import ModelArguments

logger = logging.get_logger(__name__)

from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

    _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)


# Modified from:
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/llama/modeling_llama.py


# Modified from:
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/llama/modeling_llama.py
def qwen2_flash_attention_2_forward(
        self: "Qwen2FlashAttention2",
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional["Cache"] = None,
        output_attentions: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                "with a layer index."
            )
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

    # Because the input can be padded, the absolute sequence length depends on the max position id.
    rotary_seq_len = max(kv_seq_len, position_ids[:, -1].max().item()) + 1
    cos, sin = self.rotary_emb(value_states, seq_len=rotary_seq_len)

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    use_sliding_windows = (
            _flash_supports_window_size
            and getattr(self.config, "sliding_window", None) is not None
            and kv_seq_len > self.config.sliding_window
            and self.config.use_sliding_window
    )
    print({
            "_flash_supports_window_size" : _flash_supports_window_size,
            "{getattr(self.config, 'sliding_window', None))": getattr(self.config, "sliding_window", None) is not None,
            "kv_seq_len > self.config.sliding_window": (kv_seq_len, self.config.sliding_window),
            "self.config.use_sliding_window": self.config.use_sliding_window
    })
    logger.warning_once({
            "_flash_supports_window_size" : _flash_supports_window_size,
            "{getattr(self.config, 'sliding_window', None))": getattr(self.config, "sliding_window", None) is not None,
            "kv_seq_len > self.config.sliding_window": (kv_seq_len, self.config.sliding_window),
            "self.config.use_sliding_window": self.config.use_sliding_window
    })

    if not _flash_supports_window_size:
        logger.warning_once(
            "The current flash attention version does not support sliding window attention, for a more memory "
            "efficient implementation"
            " make sure to upgrade flash-attn library."
        )
    if past_key_value is not None:
        # Activate slicing cache only if the config has a value `sliding_windows` attribute
        cache_has_contents = past_key_value.get_seq_length(self.layer_idx) > 0
        if (
                getattr(self.config, "sliding_window", None) is not None
                and kv_seq_len > self.config.sliding_window
                and cache_has_contents
        ):
            slicing_tokens = 1 - self.config.sliding_window
            print(
                f"will truncate past_key_value cache from total {kv_seq_len} tokens "
                f"to last {-slicing_tokens} tokens cache"
            )
            logger.warning_once(
                f"will truncate past_key_value cache from total {kv_seq_len} tokens "
                f"to last {-slicing_tokens} tokens cache"
            )

            past_key = past_key_value[self.layer_idx][0]
            past_value = past_key_value[self.layer_idx][1]
            print(
                f"{past_key_value[self.layer_idx][0]} shape before truncate: {past_key_value[self.layer_idx][0].shape()}"
            )
            logger.info(
                f"{past_key_value[self.layer_idx][0]} shape before truncate: {past_key_value[self.layer_idx][0].shape()}"
            )
            past_key = past_key[:, :, slicing_tokens:, :].contiguous()
            past_value = past_value[:, :, slicing_tokens:, :].contiguous()
            print(
                f"{past_key_value[self.layer_idx][0]} shape before truncate: {past_key_value[self.layer_idx][0].shape()}"
            )
            logger.info(
                f"{past_key_value[self.layer_idx][0]} shape before truncate: {past_key_value[self.layer_idx][0].shape()}"
            )

            if past_key.shape[-2] != self.config.sliding_window - 1:
                raise ValueError(
                    f"past key must have a shape of (`batch_size, num_heads, self.config.sliding_window-1, head_dim`), got"
                    f" {past_key.shape}"
                )

            if attention_mask is not None:
                attention_mask = attention_mask[:, slicing_tokens:]
                attention_mask = torch.cat([attention_mask, torch.ones_like(attention_mask[:, -1:])], dim=-1)

        cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    dropout_rate = 0.0 if not self.training else self.attention_dropout

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in float16 just to be sure everything works as expected.
    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        logger.warning_once(
            f"The input hidden states seems to be silently casted in float32, this might be related to"
            f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
            f" {target_dtype}."
        )

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    # Reashape to the expected shape for Flash Attention
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    attn_output = self._flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        q_len,
        dropout=dropout_rate,
        use_sliding_windows=use_sliding_windows,
    )

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


# Modified from:
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/llama/modeling_llama.py


def _apply_qwen_patch() -> None:
    require_version("transformers==4.41.2", "To fix: pip install transformers==4.41.2")
    logger.info("replace Qwen2FlashAttention2 with sliding window attention.")
    Qwen2FlashAttention2.forward = qwen2_flash_attention_2_forward


def configure_sliding_window_attention(config: "PretrainedConfig", model_args: "ModelArguments",
                                       is_trainable: bool) -> None:
    logger = get_logger(__name__)
    if not is_trainable or not model_args.sliding_window_attn:
        logger.info("Not Using sliding window attention.")
        return

    if getattr(config, "model_type", None) in SUPPORTED_CLASS_FOR_SWATTN:
        setattr(config, "sliding_window", 1024)
        setattr(config, "max_window_layers", 35)  # 底层感受野本来就小，上层感受野本来就大。
        setattr(config, "use_sliding_window", True)
        # 需要修改YARN和logn么？不知道
        # qwen可以直接打开sliding window
        _apply_qwen_patch()
        logger.info("Using sliding window attention with sliding_window=1024 and max_window_layers=35.")
    else:
        logger.warning("Current model does not support sliding window attention.")
