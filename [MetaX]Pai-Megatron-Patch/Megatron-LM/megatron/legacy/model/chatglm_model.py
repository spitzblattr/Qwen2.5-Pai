# !/usr/bin/env python3
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# 2024 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved. 

"""ChatGLM model."""

import torch

from megatron.training import get_args
from megatron.core import tensor_parallel
from .module import MegatronModule

from .enums import AttnMaskType
from .language_model import parallel_lm_logits
from .language_model import get_language_model
from .utils import init_method_normal
from .utils import scaled_init_method_normal


def post_language_model_processing(lm_output, labels, logit_weights,
                                   parallel_output,
                                   fp16_lm_cross_entropy):
    """
    Post language model processing
    Args:
        lm_output:
        labels:
        logit_weights:
        parallel_output:
        fp16_lm_cross_entropy:
    Return:
        loss:
    """

    # Output. Format [s b h]
    output = parallel_lm_logits(
        lm_output,
        logit_weights,
        parallel_output)

    if labels is None:
        # [s b h] => [b s h]
        return output.transpose(0, 1).contiguous()
    else:
        # [b s] => [s b]
        labels = labels.transpose(0, 1).contiguous()
        if fp16_lm_cross_entropy:
            assert output.dtype == torch.half
            loss = tensor_parallel.vocab_parallel_cross_entropy(output, labels)
        else:
            loss = tensor_parallel.vocab_parallel_cross_entropy(output.float(), labels)

        # [s b] => [b, s]
        loss = loss.transpose(0, 1).contiguous()
        return loss


class ChatGLMModel(MegatronModule):
    """ChatGLM Language model."""

    def __init__(self,
                 config,
                 num_tokentypes=0,
                 parallel_output=True,
                 pre_process=True,
                 post_process=True):
        """
        Init method
        Args:
            num_tokens:
            parallel_output:
            pre_process:
            post_process:
        Return:
        """
        args = get_args()
        super().__init__(config=config, share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights)

        # Rotary Embedding and SwiGLU are required for ChatGLM model
        if not args.use_rotary_position_embeddings:
            raise RuntimeError("Rotary embedding is required for ChatGLM model")
        if not args.swiglu:
            raise RuntimeError("SwiGLU is required for ChatGLM model")


        self.parallel_output = parallel_output
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy
        self.untie_embeddings_and_output_weights = args.untie_embeddings_and_output_weights

        self.language_model, self._language_model_key = get_language_model(
            config=config,
            num_tokentypes=num_tokentypes,
            add_pooler=False,
            encoder_attn_mask_type=AttnMaskType.causal,
            pre_process=self.pre_process,
            post_process=self.post_process
        )

        if not args.untie_embeddings_and_output_weights:
            self.initialize_word_embeddings(init_method_normal)

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        self.language_model.set_input_tensor(input_tensor)

    def forward(self, input_ids, position_ids, attention_mask,
                retriever_input_ids=None, retriever_position_ids=None, retriever_attn_mask=None,
                labels=None, tokentype_ids=None, inference_params=None):
        """
        Forward pass
        Args:
            input_ids:
            position_ids:
            attention_mask:
            ret_input_ids:
            ret_position_ids:
            ret_attn_mask:
            labels:
            tokentype_ids:
            inference_params:
        Return:
            lm_output:
        """

        lm_output = self.language_model(
            input_ids,
            position_ids,
            attention_mask,
            retriever_input_ids=retriever_input_ids,
            retriever_position_ids=retriever_position_ids,
            retriever_attn_mask=retriever_attn_mask,
            inference_params=inference_params)

        if self.post_process:
            if self.untie_embeddings_and_output_weights:
                weight = self.language_model.output_layer.weight
            else:
                weight = self.word_embeddings_weight()

            return post_language_model_processing(
                lm_output, labels,
                weight,
                self.parallel_output,
                self.fp16_lm_cross_entropy)
        else:
            return lm_output

    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):
        """
        State dict for save checkpoints
        Args:
            prefix:
            keep_vars:
        Return:
            state_dict_:
        """

        state_dict_ = {}
        state_dict_[self._language_model_key] \
            = self.language_model.state_dict_for_save_checkpoint(
                prefix=prefix, keep_vars=keep_vars)
        # Save word_embeddings.
        if self.post_process and not self.pre_process and not self.untie_embeddings_and_output_weights:
            state_dict_[self._word_embeddings_for_head_key] \
                = self.word_embeddings.state_dict(prefix=prefix,
                                                  keep_vars=keep_vars)
        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        # Load word_embeddings.
        if self.post_process and not self.pre_process and not self.untie_embeddings_and_output_weights:
            self.word_embeddings.load_state_dict(
                state_dict[self._word_embeddings_for_head_key], strict=strict)
        if self._language_model_key in state_dict:
            state_dict = state_dict[self._language_model_key]
        self.language_model.load_state_dict(state_dict, strict=strict)
