"""
# 自定义生成模型
"""
import os
import logging
import torch
from typing import Any, Dict, Iterable, List, Optional, Tuple
from transformers import (
    GPT2LMHeadModel
)
from transformers.file_utils import ModelOutput

logger = logging.getLogger(__name__)


class CustomGPTGeneration(GPT2LMHeadModel):
    """custom gpt generation model"""
    def __init__(self, config):
        super(CustomGPTGeneration, self).__init__(config)

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        """覆写父类方法"""
        # only last token for inputs_ids if past is defined in kwargs
        token_type_ids = kwargs.get("token_type_ids", None)  # 获取token type
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None

        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }

    @staticmethod
    def _update_model_kwargs_for_generation(
            outputs: ModelOutput,
            model_kwargs: Dict[str, Any],
            is_encoder_decoder: bool = False
    ) -> Dict[str, Any]:
        # update past
        if "past_key_values" in outputs:
            model_kwargs["past"] = outputs.past_key_values
        elif "mems" in outputs:
            model_kwargs["past"] = outputs.mems
        elif "past_buckets_states" in outputs:
            model_kwargs["past"] = outputs.past_buckets_states
        else:
            model_kwargs["past"] = None

        # update attention mask
        if not is_encoder_decoder:
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                # shape: [batch_size, sequence_length]
                model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )

            # update token type
            if "token_type_ids" in model_kwargs:
                token_type_ids = model_kwargs.get("token_type_ids", None)
                if token_type_ids is not None:
                    target_type_id = model_kwargs.get("target_type_id", None)  # target text type id(生成token的type id)
                    if target_type_id is None:
                        target_type_id = token_type_ids[0][-1].item() + 1  # 获取type最后一个元素+1
                        model_kwargs["target_type_id"] = target_type_id  # update dict element value
                    model_kwargs["token_type_ids"] = torch.cat(
                        [token_type_ids, token_type_ids.new_ones((token_type_ids.shape[0], 1)) * target_type_id],
                        dim=-1
                    )

        return model_kwargs

    @staticmethod
    def _expand_inputs_for_generation(
            input_ids: torch.LongTensor,
            expand_size: int = 1,
            is_encoder_decoder: bool = False,
            attention_mask: torch.LongTensor = None,
            encoder_outputs: ModelOutput = None,
            **model_kwargs
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)

        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask.index_select(0, expanded_return_idx)

        if model_kwargs["token_type_ids"] is not None:
            model_kwargs["token_type_ids"] = model_kwargs["token_type_ids"].index_select(0, expanded_return_idx)

        if is_encoder_decoder:
            assert encoder_outputs is not None
            encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(
                0, expanded_return_idx
            )
            model_kwargs["encoder_outputs"] = encoder_outputs

        return input_ids, model_kwargs
