# Copyright (c) 2023 Alibaba PAI and Nvidia Megatron-LM Team.
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

from functools import partial
import torch
import os

from megatron.core.enums import ModelType
from megatron.training.utils import get_ltor_masks_and_position_ids
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    average_losses_across_data_parallel_group
)
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training import get_args
from megatron.training import get_timers
from megatron.core import tensor_parallel
from megatron.core import mpu
from megatron.training.utils import average_losses_across_data_parallel_group, print_rank_0
from megatron.core.datasets.blended_megatron_dataset_builder_pai import BlendedMegatronDatasetBuilder
from megatron.core.datasets.utils import get_blend_from_list
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.gpt_dataset import MockGPTDataset, GPTDataset
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    average_losses_across_data_parallel_group
)
from megatron.training import pretrain


from megatron_patch.data import \
    build_pretrain_dataset_from_original, build_pretrain_dataset_from_idxmap
from megatron_patch.model.qwen.gpt_model import GPTModel
from megatron_patch.tokenizer import get_tokenizer, build_tokenizer
# from megatron_patch.training import pretrain
from megatron_patch.arguments import get_patch_args

def model_provider(pre_process=True, post_process=True):
    args = get_args()
    build_tokenizer(args)
    config = core_transformer_config_from_args(get_args())
    model = GPTModel(
        config,
        num_tokentypes=0,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process
    )
    return model

def get_batch(data_iterator):
    """Generate a batch."""

    # TODO: this is pretty hacky, find a better way
    if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
        return None, None, None, None, None

    args = get_args()

    if "-Raw" in args.dataset:
        # get batches based on the TP rank you are on
        batch = get_batch_on_this_tp_rank_original(data_iterator)
        # slice batch along sequence dimension for context parallelism
        batch = get_batch_on_this_cp_rank(batch)

    elif "-Idxmap" in args.dataset:
        # get batches based on the TP rank you are on
        batch = get_batch_on_this_tp_rank(data_iterator)
        # slice batch along sequence dimension for context parallelism
        batch = get_batch_on_this_cp_rank(batch)

    else:
        raise ValueError("please set correct --dataset ")

    return batch.values()

# def get_batch(data_iterator):
#     """Generate a batch"""
#     args = get_args()
#     tokenizer = get_tokenizer()
#     datatype = torch.int64

#     keys = ['input_ids', 'labels']
#     if data_iterator is not None:
#         data = next(data_iterator)
#     else:
#         data = None
#     data_b = tensor_parallel.broadcast_data(keys, data, datatype)
#     tokens_ = data_b['input_ids'].long()

#     labels = tokens_[:, 1:].contiguous()
#     tokens = tokens_[:, :-1].contiguous()

#     # Get the masks and postition ids.
#     attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
#         tokens,
#         tokenizer.pad_token_id,
#         args.reset_position_ids,
#         args.reset_attention_mask,
#         True)
#     return tokens, labels, loss_mask, attention_mask, position_ids

def loss_func(loss_mask, output_tensor):
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss, {'lm loss': averaged_loss[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
        data_iterator)
    timers('batch-generator').stop()
    output_tensor = model(tokens, position_ids, attention_mask,
                          labels=labels)

    return output_tensor, partial(loss_func, loss_mask)


def is_dataset_built_on_rank():
    return (mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()) and mpu.get_tensor_model_parallel_rank() == 0


def core_gpt_dataset_config_from_args(args):
    tokenizer = get_tokenizer()

    return GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=get_blend_from_list(args.data_path),
        blend_per_split=[
            get_blend_from_list(args.train_data_path),
            get_blend_from_list(args.valid_data_path),
            get_blend_from_list(args.test_data_path)
        ],
        split=args.split,
        num_dataset_builder_threads=args.num_dataset_builder_threads,
        path_to_cache=args.data_cache_path,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
    )


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()
    print_rank_0("> building train, validation, and test datasets for GPT ...")

    if "-Raw" in args.dataset:
        train_ds, valid_ds, test_ds = build_pretrain_dataset_from_original(args.dataset)
    else:
        config = core_gpt_dataset_config_from_args(args)

        if config.mock:
            dataset_type = MockGPTDataset
        else:
            dataset_type = GPTDataset
        train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
            dataset_type,
            train_val_test_num_samples,
            is_dataset_built_on_rank,
            config
        ).build()

        print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds

if __name__ == "__main__":

    pretrain(train_valid_test_datasets_provider, model_provider,
             ModelType.encoder_or_decoder,
             forward_step,
             extra_args_provider=get_patch_args)