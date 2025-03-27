# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain GPT."""

import os
import torch
from functools import partial
from typing import Union
from megatron.training import get_args
from megatron.training import print_rank_0
from megatron.training import get_timers
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
#from megatron.core.datasets.blended_megatron_dataset_builder_pai import BlendedMegatronDatasetBuilder
from megatron.core.datasets.utils import get_blend_from_list
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.gpt_dataset import MockGPTDataset, GPTDataset
import megatron.legacy.model
from megatron.training import pretrain
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    average_losses_across_data_parallel_group
)
from megatron.training.arguments import core_transformer_config_from_args

from megatron_patch.data import train_valid_test_datasets_provider
# from megatron_patch.model.qwen2.layer_specs import get_gpt_layer_with_transformer_engine_spec,get_gpt_layer_local_spec
from megatron_patch.model.qwen2.layer_specs import get_gpt_layer_local_spec
from megatron_patch.model.qwen2.model import GPTModel
from megatron_patch.model.qwen2.transformer_config import Qwen2TransformerConfig
from megatron_patch.arguments import get_patch_args
from megatron_patch.tokenizer import get_tokenizer, build_tokenizer
import torch._dynamo
torch._dynamo.config.suppress_errors = True

def model_provider(
    pre_process=True, post_process=True
) -> Union[GPTModel]:

    args = get_args()

    build_tokenizer(args)
    print_rank_0("building qwen2 model ...")

    config = core_transformer_config_from_args(args, Qwen2TransformerConfig)
    use_te = args.transformer_impl == "transformer_engine"

    if use_te:
        print_rank_0("building qwen2 model in TE...")
        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            args.num_experts, args.moe_grouped_gemm, args.qk_layernorm
        )
    else:
        print_rank_0("building qwen2 model in Mcore...")
        transformer_layer_spec = get_gpt_layer_local_spec(
            args.num_experts, args.moe_grouped_gemm, args.qk_layernorm
        )

    model = GPTModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
        seq_len_interpolation_factor=args.rotary_seq_len_interpolation_factor,
    )

    return model


if __name__ == "__main__":
    from megatron_patch.template.helper import forward_step
    from megatron_patch.data import train_valid_test_datasets_provider
    train_valid_test_datasets_provider.is_distributed = True

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        extra_args_provider=get_patch_args,
    )