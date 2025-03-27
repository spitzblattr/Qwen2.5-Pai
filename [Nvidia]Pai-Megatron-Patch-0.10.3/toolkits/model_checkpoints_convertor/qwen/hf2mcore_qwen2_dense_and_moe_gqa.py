import os
import re
import json
import torch
import copy
import safetensors
from collections import defaultdict
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)

from megatron.training.initialize import initialize_megatron
from megatron.training import get_args
from megatron.training.checkpointing import get_checkpoint_name, get_checkpoint_tracker_filename, read_metadata
from functools import partial
from megatron.training.utils import get_ltor_masks_and_position_ids
import sys

path_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
sys.path.append(os.path.join(path_dir, "examples"))
from qwen2.pretrain_qwen import model_provider
from megatron_patch.arguments import get_patch_args

from toolkits.model_checkpoints_convertor.utils import (
    save_state_dict,
    save_hfmodel
)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

def add_model_args(parser):

    parser.add_argument(
        "--target-tensor-model-parallel-size",
        type=int,
        default=1
    )

    parser.add_argument(
        "--target-pipeline-model-parallel-size",
        type=int,
        default=1
    )

    parser.add_argument(
        "--target-expert-model-parallel-size",
        type=int,
        default=1
    )

    parser.add_argument(
        "--hf-ckpt-path",
        type=str
    )

    parser.add_argument(
        "--save-safetensors",
        action='store_false',
    )

    return parser


def name_to_expert_rank(key):
    pattern = r'local_experts\.(\d+)\.'
    expert_rank = int(re.findall(pattern, key)[0])
    return expert_rank

def load_megatron_model(args):
    args.tensor_model_parallel_size = args.target_tensor_model_parallel_size
    args.pipeline_model_parallel_size = args.target_pipeline_model_parallel_size

    if args.num_experts is not None:
        args.expert_model_parallel_size = args.target_expert_model_parallel_size

    if args.tensor_model_parallel_size >1:
        args.sequence_parallel = True

    assert args.num_query_groups >= args.target_tensor_model_parallel_size

    os.makedirs(args.save, exist_ok=True)
    # 从 args.hf_ckpt_path 复制 Hugging Face 预训练模型的配置和词汇文件到 args.save 和 args.load 目录
    os.system("cp -rf " + args.hf_ckpt_path + "/config*.json " + args.save)
    os.system("cp -rf " + args.hf_ckpt_path + "/generation_config.json " + args.save)
    os.system("cp -rf " + args.hf_ckpt_path+ "/tokenizer* " + args.save)
    os.system("cp -rf " + args.hf_ckpt_path + "/vocab.json " + args.save)
    os.system("cp -rf " + args.hf_ckpt_path + "/merges.txt " + args.save)

    os.system("cp -rf " + args.hf_ckpt_path + "/config*.json " + args.load)
    os.system("cp -rf " + args.hf_ckpt_path + "/generation_config.json " + args.load)
    os.system("cp -rf " + args.hf_ckpt_path+ "/tokenizer* " + args.load)
    os.system("cp -rf " + args.hf_ckpt_path + "/vocab.json " + args.load)
    os.system("cp -rf " + args.hf_ckpt_path + "/merges.txt " + args.load)

    model = model_provider()

    model_path = args.load
    # 从 args.load 里的 'latest_checkpointed_iteration.txt' 里获取最新的 checkpoint 文件名
    tracker_filename = get_checkpoint_tracker_filename(model_path)
    iteration, release = read_metadata(tracker_filename)
    head_dim = args.hidden_size // args.num_attention_heads
    group_per_split = args.num_query_groups // args.tensor_model_parallel_size
    if args.num_experts is not None:
        num_local_experts = args.num_experts // args.expert_model_parallel_size
    state_dict = {}
    mid_state = defaultdict(list)
    if (
        args.tensor_model_parallel_size == 1
        and args.pipeline_model_parallel_size == 1
    ):
        checkpoint_name = get_checkpoint_name(model_path, iteration, release, None, None, None, None, None)
        # checkpoint_name = /home/train_finished_models/iter_0024000/mp_rank_00/model_optim_rng.pt
        state_dict = torch.load(checkpoint_name, weights_only=False)['model']

    elif (
        args.tensor_model_parallel_size > 1
        and args.pipeline_model_parallel_size == 1
        and args.num_experts is None
    ):  
        for tp_rank in range(args.tensor_model_parallel_size):
            checkpoint_name = get_checkpoint_name(model_path, iteration, release, None, tp_rank, None, None, None)
            print(f'load {checkpoint_name}')
            split_state = torch.load(checkpoint_name, map_location="cpu", weights_only=False)['model']
            for k, v in split_state.items():
                mid_state[k].append(v)
        for k, v in mid_state.items():
            if not isinstance(v[0], torch.Tensor) or 'norm' in k:
                target_v = v[0]
            elif 'extra_state' in k:
                target_v = None
            elif 'embedding' in k or 'output_layer' in k:
                target_v = torch.cat(v, dim=0)
            elif 'linear_proj' in k or 'linear_fc2' in k:
                target_v = torch.cat(v, dim=1)
            elif 'linear_qkv.weight' in k:
                viewed = [x.view(group_per_split, -1, head_dim, args.hidden_size) for x in v]
                target_v = torch.cat(viewed, dim=0).view(-1, args.hidden_size)
            elif 'linear_qkv.bias' in k:
                viewed = [x.view(group_per_split, -1) for x in v]
                target_v = torch.cat(viewed, dim=0).view(-1)
            elif 'linear_fc1' in k:
                viewed = [x.view(2, -1, args.hidden_size) for x in v]
                target_v = torch.cat(viewed, dim=1).view(-1, args.hidden_size)
            else:
                raise ValueError
            state_dict[k] = target_v
    elif (
        args.tensor_model_parallel_size > 1
        and args.pipeline_model_parallel_size > 1
        and args.num_experts is None
    ):  
        num_layers = args.num_layers // args.pipeline_model_parallel_size
        layers_to_copy = {}
        for tp_rank in range(args.tensor_model_parallel_size):
            for pp_rank in range(args.pipeline_model_parallel_size):
                layer_offset = pp_rank * num_layers
                for layer in range(num_layers):
                    pp_layer_id = layer + layer_offset
                    layers_to_copy[f"decoder.layers.{layer}"] = pp_layer_id
                checkpoint_name = get_checkpoint_name(model_path, iteration, release, True, tp_rank, pp_rank, None, None)
                print(f'load {checkpoint_name}')
                split_state = torch.load(checkpoint_name, map_location="cpu", weights_only=False)['model']
                for k, v in split_state.items():
                    try:
                        pattern = re.compile(r'\d+')
                        res = pattern.findall(k)
                        k = re.sub(r"decoder.layers.\d+", "decoder.layers." + str(layers_to_copy["decoder.layers." + res[0]]), k)
                        mid_state[k].append(v)
                    except:
                        mid_state[k].append(v)
        for k, v in mid_state.items():
            if not isinstance(v[0], torch.Tensor) or 'norm' in k:
                target_v = v[0]
            elif 'extra_state' in k:
                target_v = None
            elif 'embedding' in k or 'output_layer' in k:
                target_v = torch.cat(v, dim=0)
            elif 'linear_proj' in k or 'linear_fc2' in k:
                target_v = torch.cat(v, dim=1)
            elif 'linear_qkv.weight' in k:
                viewed = [x.view(group_per_split, -1, head_dim, args.hidden_size) for x in v]
                target_v = torch.cat(viewed, dim=0).view(-1, args.hidden_size)
            elif 'linear_qkv.bias' in k:
                viewed = [x.view(group_per_split, -1) for x in v]
                target_v = torch.cat(viewed, dim=0).view(-1)
            elif 'linear_fc1' in k:
                viewed = [x.view(2, -1, args.hidden_size) for x in v]
                target_v = torch.cat(viewed, dim=1).view(-1, args.hidden_size)
            else:
                raise ValueError
            state_dict[k] = target_v

    elif (
        args.tensor_model_parallel_size == 1
        and args.pipeline_model_parallel_size == 1
        and args.expert_model_parallel_size >1
        and args.num_experts % args.expert_model_parallel_size == 0
    ):
        for ep_rank in range(args.expert_model_parallel_size):
            checkpoint_name = get_checkpoint_name(model_path, iteration, release, None, None, None, True, ep_rank)
            print(f'load {checkpoint_name}')
            split_state = torch.load(checkpoint_name, map_location="cpu", weights_only=False)['model']
            for k, v in split_state.items():
                if 'local_experts' in k:
                    expert_local_rank = name_to_expert_rank(k)
                    expert_rank = expert_local_rank + num_local_experts * ep_rank
                    k = k.replace(f'local_experts.{expert_local_rank}', f'local_experts.{expert_rank}')
                state_dict[k] = v
    elif (
        args.tensor_model_parallel_size > 1
        and args.pipeline_model_parallel_size == 1
        and args.expert_model_parallel_size > 1
        and args.num_experts % args.expert_model_parallel_size == 0
    ):
        for tp_rank in range(args.tensor_model_parallel_size):
            for ep_rank in range(args.expert_model_parallel_size):
                if args.expert_model_parallel_size >1:
                    checkpoint_name = get_checkpoint_name(model_path, iteration,release, None, tp_rank, None, True, ep_rank)
                elif args.expert_model_parallel_size ==1:
                    checkpoint_name = get_checkpoint_name(model_path, iteration, release, None, tp_rank, None, False)
                print(f'load {checkpoint_name}')
                split_state = torch.load(checkpoint_name, map_location="cpu", weights_only=False)['model']
                for k, v in split_state.items():
                    if 'local_experts' in k and 'norm' not in k:
                        local_expert_rank = name_to_expert_rank(k)
                        expert_rank = local_expert_rank + num_local_experts * ep_rank
                        k = k.replace(f'local_experts.{local_expert_rank}', f'local_experts.{expert_rank}')
                        mid_state[k].append(v)
                    elif ep_rank == 0:
                        mid_state[k].append(v)

        for k, v in mid_state.items():
            if not isinstance(v[0], torch.Tensor) or 'router' in k or 'gate' in k:
                target_v = v[0]
            elif 'extra_state' in k:
                target_v = None
            elif 'embedding' in k or 'output_layer' in k:
                target_v = torch.cat(v, dim=0)
            elif 'linear_proj' in k or 'linear_fc2' in k:
                target_v = torch.cat(v, dim=1)
            elif 'linear_qkv.weight' in k:
                viewed = [x.view(group_per_split, -1, head_dim, args.hidden_size) for x in v]
                target_v = torch.cat(viewed, dim=0).view(-1, args.hidden_size)
            elif 'linear_qkv.bias' in k:
                viewed = [x.view(group_per_split, -1) for x in v]
                target_v = torch.cat(viewed, dim=0).view(-1)
            elif 'linear_fc1' in k:
                viewed = [x.view(2, -1, args.hidden_size) for x in v]
                target_v = torch.cat(viewed, dim=1).view(-1, args.hidden_size)
            elif 'input_layernorm' in k:
                target_v = v[0]
            elif 'kv_layernorm' in k:
                target_v = v[0]
            elif 'pre_mlp_layernorm' in k:
                target_v = v[0]
            else:
                print(f"Missing {k}")
                raise ValueError
            state_dict[k] = target_v

    elif (
        args.tensor_model_parallel_size >= 1
        and args.pipeline_model_parallel_size > 1
        and args.expert_model_parallel_size >= 1
        and ( args.num_experts is None or args.num_experts % args.expert_model_parallel_size == 0 )
    ):
        num_layers = args.num_layers // args.pipeline_model_parallel_size
        layers_to_copy = {}
        for tp_rank in range(args.tensor_model_parallel_size):
            for ep_rank in range(args.expert_model_parallel_size):
                for pp_rank in range(args.pipeline_model_parallel_size):
                    layer_offset = pp_rank * num_layers
                    for layer in range(num_layers):
                        pp_layer_id = layer + layer_offset
                        layers_to_copy[f"decoder.layers.{layer}"] = pp_layer_id

                    if args.expert_model_parallel_size > 1:
                        checkpoint_name = get_checkpoint_name(model_path, iteration, release, True, tp_rank, pp_rank, True,
                                                              ep_rank)
                    elif args.expert_model_parallel_size == 1:
                        checkpoint_name = get_checkpoint_name(model_path, iteration, release, True, tp_rank, pp_rank,
                                                              False)
                    print(f'load {checkpoint_name}')
                    split_state = torch.load(checkpoint_name, map_location="cpu", weights_only=False)['model']
                    for k, v in split_state.items():
                        try:
                            if 'local_experts' in k:
                                local_expert_rank = name_to_expert_rank(k)
                                expert_rank = local_expert_rank + num_local_experts * ep_rank
                                k = k.replace(f'local_experts.{local_expert_rank}', f'local_experts.{expert_rank}')
                            pattern = re.compile(r'\d+')
                            res = pattern.findall(k)
                            tgt = re.sub(r"decoder.layers.\d+","decoder.layers." + str(layers_to_copy["decoder.layers." + res[0]]), k)
                            if 'linear_proj' in k or 'linear_q_proj' in k or 'linear_kv_up_proj' in k or 'decoder.layers.0.mlp.linear_fc2' in k or \
                                    'decoder.layers.0.mlp.linear_fc1' in k or 'shared_experts.linear_fc1' in k or 'shared_experts.linear_fc2' in k:
                                if ep_rank ==0:
                                    mid_state[tgt].append(v)
                            else:
                                mid_state[tgt].append(v)
                        except:
                            print(f"Skipping {k}")
                            if "word_embeddings" in k:
                                if ep_rank ==0 and pp_rank == 0:
                                    mid_state[k].append(v)
                            elif "output_layer" in k or "final_layernorm" in k:
                                if ep_rank ==0 and pp_rank == args.pipeline_model_parallel_size - 1:
                                    mid_state[k].append(v)
                            else:
                                raise ValueError("Something is wrong!")
        for k, v in mid_state.items():
            if not isinstance(v[0], torch.Tensor) or 'router' in k or 'gate' in k:
                target_v = v[0]
            elif 'extra_state' in k:
                target_v = None
            elif 'word_embeddings' in k or 'output_layer' in k or 'final_layernorm' in k:
                target_v = torch.cat(v, dim=0)
            elif 'linear_proj' in k or 'linear_fc2' in k:
                target_v = torch.cat(v, dim=1)
            elif 'linear_qkv.weight' in k:
                viewed = [x.view(group_per_split, -1, head_dim, args.hidden_size) for x in v]
                target_v = torch.cat(viewed, dim=0).view(-1, args.hidden_size)
            elif 'linear_qkv.bias' in k:
                viewed = [x.view(group_per_split, -1) for x in v]
                target_v = torch.cat(viewed, dim=0).view(-1)
            elif 'linear_fc1' in k and "layer_norm_weight" not in k:
                viewed = [x.view(2, -1, args.hidden_size) for x in v]
                target_v = torch.cat(viewed, dim=1).view(-1, args.hidden_size)
            elif 'input_layernorm' in k:
                target_v = v[0]
            elif 'layer_norm_weight' in k:
                target_v = v[0]
            elif 'pre_mlp_layernorm' in k:
                target_v = v[0]
            else:
                print(f"Missing {k}")
                raise ValueError
            state_dict[k] = target_v

    else:
        raise ValueError('not support yet')

    model.load_state_dict(state_dict, strict=False)
    return model


def convert_checkpoint_from_megatron_to_transformers(mgmodel, hfmodel, args):

    # 先根据 args.fp16 或 args.bf16，把源模型和目标模型都设置成相应的精度
    if args.fp16:
        mgmodel = mgmodel.half()
        hfmodel = hfmodel.half()
    elif args.bf16:
        mgmodel = mgmodel.bfloat16()
        hfmodel = hfmodel.bfloat16()

    # 以下参数主要用于后面把合并在一起的 QKV 权重拆成 Q、K、V 各自部分时，所需的 shape 计算
    num_query_groups = args.num_query_groups
    hidden_size = args.hidden_size
    head_dim = hidden_size // args.num_attention_heads
    # 是否使用 NVIDIA 的 Transformer Engine 实现。若为 True，后面的 LayerNorm 等权重复制逻辑会有所区别
    use_te = args.transformer_impl == "transformer_engine"
    # 每个 query group 下分配多少个注意力头（针对 V 部分）
    value_num_per_group = args.num_attention_heads // num_query_groups
    q_dim_per_group = hidden_size // num_query_groups
    kv_dim_per_group = head_dim
    with torch.no_grad():
        # HuggingFace 模型和 Megatron 模型的词向量部分都有一张 weight 矩阵，表示 vocab_size x hidden_size 的词嵌入。
        # 直接把 Megatron 的 word_embeddings 复制到 HF 模型的 embed_tokens。它们的形状通常是一致的。
        hfmodel.model.embed_tokens.weight.copy_(mgmodel.embedding.word_embeddings.weight)
        # Megatron 中的 mgmodel.decoder.layers 跟 HF 中的 hfmodel.model.layers 都是一个列表，包含多个 Transformer 子层。
        # 通过 zip(...) 一一对应地取出，每一对 (mglayer, hflayer) 就表示同一层的参数需要相互对应。
        for mglayer, hflayer in zip(mgmodel.decoder.layers, hfmodel.model.layers):
            # 根据 use_te 不同，这里选择复制不同位置的权重。
            # 若使用了 transformer_engine，Megatron 可能把输入层的 LayerNorm 集成在了 linear_qkv.layer_norm_weight；否则就是 input_layernorm.weight。
            if use_te:
                hflayer.input_layernorm.weight.copy_(mglayer.self_attention.linear_qkv.layer_norm_weight)
            else:
                hflayer.input_layernorm.weight.copy_(mglayer.input_layernorm.weight)

            ''' 拆分并复制自注意力中的 QKV weight'''
            # 在 Megatron 中，Q、K、V 通常是合并在一个大矩阵 linear_qkv.weight 里，然后在 forward 中再做拆分。
            # 这里先 view(num_query_groups, -1, head_dim, hidden_size)，再用 torch.split 把这个大矩阵拆分成 3 个部分：Q 的那部分、K 的那部分、V 的那部分。
            # 不同的切分大小由 [value_num_per_group, 1, 1] 决定，是因为在多查询/分组注意力下，每个组可能有 value_num_per_group 个 head，用于 Q，而 K、V 只拆分出 1 份（或者在普通多头情况下，这里的切分相当于把 Q、K、V 平均拆开）。
            # 最后 reshape(-1, hidden_size)，得到和 HuggingFace 里 q_proj.weight 等层期望的形状对应，然后调用 copy_ 完成复制。            
            qkv_weight = mglayer.self_attention.linear_qkv.weight.view(num_query_groups, -1, head_dim, hidden_size)
            q_weight, k_weight, v_weight = torch.split(qkv_weight, split_size_or_sections=[value_num_per_group, 1, 1], dim=1)
            hflayer.self_attn.q_proj.weight.copy_(q_weight.reshape(-1, hidden_size))
            hflayer.self_attn.k_proj.weight.copy_(k_weight.reshape(-1, hidden_size))
            hflayer.self_attn.v_proj.weight.copy_(v_weight.reshape(-1, hidden_size))

            ''' 拆分并复制自注意力中的 QKV bias'''
            # 和 QKV 的权重类似，Megatron 把偏置也合并成一个整体 bias，这里再按 [q_dim_per_group, kv_dim_per_group, kv_dim_per_group] 拆分。
            # 因为 Q 的维度和 K、V 的维度在多查询场景下不一定相同，所以要用 q_dim_per_group、kv_dim_per_group 来拆分。
            # 最后把拆好的 q_bias, k_bias, v_bias 复制给 HuggingFace 的 q_proj.bias, k_proj.bias, v_proj.bias。            
            qkv_bias = mglayer.self_attention.linear_qkv.bias.view(num_query_groups, -1)
            q_bias, k_bias, v_bias = torch.split(qkv_bias, split_size_or_sections=[q_dim_per_group, kv_dim_per_group, kv_dim_per_group], dim=1)
            q_bias = q_bias.contiguous().view(-1)
            k_bias = k_bias.contiguous().view(-1)
            v_bias = v_bias.contiguous().view(-1)

            hflayer.self_attn.q_proj.bias.copy_(q_bias)
            hflayer.self_attn.k_proj.bias.copy_(k_bias)
            hflayer.self_attn.v_proj.bias.copy_(v_bias)

            # 自注意力部分的最后一个线性层 o_proj 的权重。Megatron 中叫 linear_proj.weight，直接复制即可
            hflayer.self_attn.o_proj.weight.copy_(mglayer.self_attention.linear_proj.weight)

            '''复制 MLP（前馈层）相关权重：区分普通 FFN 和多专家 (MoE)'''
            if args.num_experts is None:
                # 没有使用专家网络,是一个普通的前馈层(FFN)。
                # 这里的 linear_fc1.weight 可能是一个「gate」和「FC1」合并的矩阵，因此先用 torch.split 拆出 gate_weight 和真正的 fc1_weight。
                # 分别复制给 HF 模型的 gate_proj.weight 和 up_proj.weight，并把 Megatron 中的 linear_fc2.weight 复制给 HF 的 down_proj.weight。
                gate_weight, fc1_weight = torch.split(mglayer.mlp.linear_fc1.weight, split_size_or_sections=args.ffn_hidden_size)
                hflayer.mlp.gate_proj.weight.copy_(gate_weight)
                hflayer.mlp.up_proj.weight.copy_(fc1_weight)
                hflayer.mlp.down_proj.weight.copy_(mglayer.mlp.linear_fc2.weight)
            else:
                # 如果 args.num_experts 非空
                # 先把 MoE 的「router」权重复制到 HF 的 mlp.gate；然后遍历所有专家(local_experts)的线性层，分别拆分出 gate 部分和 up 部分，再复制给 HF 对应的 hfexpert.gate_proj / hfexpert.up_proj。
                # 同理把专家的第二线性层 (linear_fc2) 复制给 down_proj。
                # 如果还有共享专家 (shared_expert)，也同样拆分复制过去。
                hflayer.mlp.gate.weight.copy_(mglayer.mlp.router.weight)
                for mgexpert, hfexpert in zip(mglayer.mlp.experts.local_experts, hflayer.mlp.experts):
                    gate_weight, up_weight = torch.split(mgexpert.linear_fc1.weight,
                                                         split_size_or_sections=args.moe_ffn_hidden_size)
                    hfexpert.gate_proj.weight.copy_(gate_weight)
                    hfexpert.up_proj.weight.copy_(up_weight)
                    hfexpert.down_proj.weight.copy_(mgexpert.linear_fc2.weight)

                hflayer.mlp.shared_expert_gate.weight.copy_(mglayer.mlp.shared_expert_gate.weight)
                shared_expert_gate_weight, shared_expert_up_weight = \
                    torch.split(mglayer.mlp.shared_expert.linear_fc1.weight,
                                split_size_or_sections=args.shared_moe_ffn_hidden_size)
                hflayer.mlp.shared_expert.gate_proj.weight.copy_(shared_expert_gate_weight)
                hflayer.mlp.shared_expert.up_proj.weight.copy_(shared_expert_up_weight)
                hflayer.mlp.shared_expert.down_proj.weight.copy_(mglayer.mlp.shared_expert.linear_fc2.weight)

            # 复制 post_attention_layernorm weight
            if use_te and not args.num_experts:
                hflayer.post_attention_layernorm.weight.copy_(mglayer.mlp.linear_fc1.layer_norm_weight)
            else:
                hflayer.post_attention_layernorm.weight.copy_(mglayer.pre_mlp_layernorm.weight)

        hfmodel.model.norm.weight.copy_(mgmodel.decoder.final_layernorm.weight)
        if args.untie_embeddings_and_output_weights:
            hfmodel.lm_head.weight.copy_(mgmodel.output_layer.weight)
        else:   
            # 如果没有指定 args.untie_embeddings_and_output_weights，并不是说模型就没有lm_head，只是表示lm_head和embed_tokens权重一样，无需独立赋值。
            # 如果模型 config 里没有 untie_embeddings_and_output_weights 之类的指定，HuggingFace 就会将模型的lm_head和embed_tokens共享权重，这就是为什么我们在网站上下载的原始 qwen2.5-0.5B 模型没有"lm_head"。
            # 因此，在该种情形下，转换为 huggingface 格式的模型也不需要显式具有 lm_head 层(否则转换后的模型体积会比微调前的原始模型大，而不是一致)。
            del hfmodel.lm_head   


def convert_checkpoint_from_transformers_to_megatron(hfmodel, mgmodel, args):

    if args.fp16:
        mgmodel = mgmodel.half()
        hfmodel = hfmodel.half()
    elif args.bf16:
        mgmodel = mgmodel.bfloat16()
        hfmodel = hfmodel.bfloat16()

    assert args.num_query_groups >= args.target_tensor_model_parallel_size

    num_attention_heads = args.num_attention_heads
    num_query_groups = args.num_query_groups
    hidden_size = args.hidden_size
    head_dim = hidden_size // num_attention_heads
    use_te = args.transformer_impl == "transformer_engine"

    with torch.no_grad():
        mgmodel.embedding.word_embeddings.weight.copy_(hfmodel.model.embed_tokens.weight)
        for mglayer, hflayer in zip(mgmodel.decoder.layers, hfmodel.model.layers):
            if use_te:
                mglayer.self_attention.linear_qkv.layer_norm_weight.copy_(hflayer.input_layernorm.weight)
            else:
                mglayer.input_layernorm.weight.copy_(hflayer.input_layernorm.weight)

            q_proj_weight = hflayer.self_attn.q_proj.weight.view(num_query_groups, -1, head_dim, hidden_size)
            k_proj_weight = hflayer.self_attn.k_proj.weight.view(num_query_groups, -1, head_dim, hidden_size)
            v_proj_weight = hflayer.self_attn.v_proj.weight.view(num_query_groups, -1, head_dim, hidden_size)
            qkv_proj = torch.cat([q_proj_weight, k_proj_weight, v_proj_weight], dim=1).view(-1, hidden_size).contiguous()
            mglayer.self_attention.linear_qkv.weight.copy_(qkv_proj)

            q_proj_bias = hflayer.self_attn.q_proj.bias.view(num_query_groups, -1)
            k_proj_bias = hflayer.self_attn.k_proj.bias.view(num_query_groups, -1)
            v_proj_bias = hflayer.self_attn.v_proj.bias.view(num_query_groups, -1)
            qkv_bias = torch.cat([q_proj_bias, k_proj_bias, v_proj_bias], dim=1).view(-1).contiguous()
            mglayer.self_attention.linear_qkv.bias.copy_(qkv_bias)

            mglayer.self_attention.linear_proj.weight.copy_(hflayer.self_attn.o_proj.weight)

            if args.num_experts is None:
                fc1_weight = torch.cat([hflayer.mlp.gate_proj.weight, hflayer.mlp.up_proj.weight])
                mglayer.mlp.linear_fc1.weight.copy_(fc1_weight)
                mglayer.mlp.linear_fc2.weight.copy_(hflayer.mlp.down_proj.weight)
            else:
                mglayer.mlp.router.weight.copy_(hflayer.mlp.gate.weight)
                for hf_expert, expert in zip(hflayer.mlp.experts, mglayer.mlp.experts.local_experts):
                    fc1_weight = torch.cat([hf_expert.gate_proj.weight, hf_expert.up_proj.weight])
                    expert.linear_fc1.weight.copy_(fc1_weight)
                    expert.linear_fc2.weight.copy_(hf_expert.down_proj.weight)
                mglayer.mlp.shared_expert_gate.weight.copy_(hflayer.mlp.shared_expert_gate.weight)
                shared_fc1_weight = torch.cat(
                    [hflayer.mlp.shared_expert.gate_proj.weight, hflayer.mlp.shared_expert.up_proj.weight])
                mglayer.mlp.shared_expert.linear_fc1.weight.copy_(shared_fc1_weight)
                mglayer.mlp.shared_expert.linear_fc2.weight.copy_(hflayer.mlp.shared_expert.down_proj.weight)

            if use_te and not args.num_experts:
                mglayer.mlp.linear_fc1.layer_norm_weight.copy_(hflayer.post_attention_layernorm.weight)
            else:
                mglayer.pre_mlp_layernorm.weight.copy_(hflayer.post_attention_layernorm.weight)

        mgmodel.decoder.final_layernorm.weight.copy_(hfmodel.model.norm.weight)
        if args.untie_embeddings_and_output_weights:
            mgmodel.output_layer.weight.copy_(hfmodel.lm_head.weight)


def check_layer(layers_to_copy, k):
    pattern = re.compile(r"decoder.layers.\d+")
    res = pattern.findall(k)
    return res and res[0] in layers_to_copy.keys()

def save_mgmodel(mgmodel, args):

    args.tensor_model_parallel_size = args.target_tensor_model_parallel_size
    args.pipeline_model_parallel_size = args.target_pipeline_model_parallel_size

    if args.num_experts is not None:
        args.expert_model_parallel_size = args.target_expert_model_parallel_size

    os.makedirs(args.save, exist_ok=True)
    os.system("cp -rf " + args.load + "/config*.json " + args.save)
    os.system("cp -rf " + args.load + "/tokenizer* " + args.save)
    os.system("cp -rf " + args.load + "/vocab.json " + args.save)
    os.system("cp -rf " + args.load + "/merges.txt " + args.save)

    tracker_filepath = os.path.join(args.save, 'latest_checkpointed_iteration.txt')
    with open(tracker_filepath, "w") as f:
        f.write("release")

    head_dim = args.hidden_size // args.num_attention_heads
    group_per_split = args.num_query_groups // args.target_tensor_model_parallel_size
    full_model = mgmodel.state_dict_for_save_checkpoint()
    num_layers = args.num_layers // args.pipeline_model_parallel_size
    for k in list(full_model.keys()):
        if 'extra_state' in k:
            # NOTE: since TE 1.14, fp8 metadata will be saved as tensor. 
            # Always drop these values in the MG ckpt to avoid potential issue.
            # This should work fine because fp8 metadata is not supported by HF ckpt.
            full_model[k] = None
        elif full_model[k] is None:
            full_model.pop(k)

    if args.num_experts is not None:
        pattern = r'local_experts\.(\d+)\.'
        num_local_experts = args.num_experts // args.expert_model_parallel_size if args.num_experts else 0

    if (
        args.tensor_model_parallel_size == 1
        and args.pipeline_model_parallel_size == 1
        and args.expert_model_parallel_size == 1
    ):
        checkpoint_name = get_checkpoint_name(args.save, 0, True)
        save_state_dict(args, [full_model], checkpoint_name)
    elif (
        args.tensor_model_parallel_size == 1
        and args.pipeline_model_parallel_size == 1
        and args.expert_model_parallel_size >1
        and args.num_experts % args.expert_model_parallel_size == 0
    ):

        for ep_rank in range(args.expert_model_parallel_size):
            model_split = {}
            checkpoint_name = get_checkpoint_name(args.save, 0, True, None, None, None, True, ep_rank)
            print(f'save ep_rank {ep_rank} model to {checkpoint_name}')
            for k, v in full_model.items():
                if 'local_experts' in k:
                    expert_rank = int(re.findall(pattern, k)[0])
                    if expert_rank // num_local_experts != ep_rank:
                        continue
                    expert_local_rank = expert_rank % num_local_experts
                    k = k.replace(f'local_experts.{expert_rank}', f'local_experts.{expert_local_rank}')
                model_split[k] = v
            save_state_dict(args, [model_split], checkpoint_name)
    elif (
        args.tensor_model_parallel_size > 1
        and args.pipeline_model_parallel_size == 1
        and args.num_experts is None
    ):
        for tp_rank in range(args.tensor_model_parallel_size):
            model_split = {}
            checkpoint_name = get_checkpoint_name(args.save, 0, True, None, tp_rank)
            print(f'tensor_parallel, save model to {checkpoint_name}')
            for k, v in full_model.items():
                if not isinstance(v, torch.Tensor):
                    target_v = v
                elif 'linear_qkv.weight' in k:
                    viewed = v.view(args.num_query_groups, -1, head_dim, args.hidden_size)
                    viewed = viewed[group_per_split*tp_rank : group_per_split*(tp_rank + 1)]
                    target_v = viewed.view(-1, args.hidden_size)
                elif 'linear_qkv.bias' in k:
                    viewed = v.view(args.num_query_groups, -1, head_dim)
                    viewed = viewed[group_per_split * tp_rank: group_per_split * (tp_rank + 1)]
                    target_v = viewed.view(-1)
                elif 'linear_proj' in k or 'linear_fc2' in k:
                    seg = v.shape[1] // args.tensor_model_parallel_size
                    target_v = v[:, seg*tp_rank : seg*(tp_rank + 1)]
                elif 'embedding' in k or 'output_layer' in k:
                    seg = v.shape[0] // args.tensor_model_parallel_size
                    target_v = v[seg*tp_rank : seg*(tp_rank + 1)]
                elif 'linear_fc1' in k and 'norm' not in k:
                    viewed = v.view(-1, args.ffn_hidden_size, args.hidden_size)
                    seg = args.ffn_hidden_size // args.tensor_model_parallel_size
                    target_v = viewed[:, seg*tp_rank: seg*(tp_rank+1), :].reshape(-1, args.hidden_size)
                else:
                    target_v = v
                model_split[k] = target_v
            save_state_dict(args, [model_split], checkpoint_name)
    elif (
        args.tensor_model_parallel_size > 1
        and args.pipeline_model_parallel_size == 1
        and args.num_experts % args.expert_model_parallel_size == 0
    ):
        for tp_rank in range(args.tensor_model_parallel_size):
            for ep_rank in range(args.expert_model_parallel_size):
                model_split = {}
                if args.expert_model_parallel_size >1:
                    checkpoint_name = get_checkpoint_name(args.save, 0, True, None, tp_rank, None, True, ep_rank)
                elif args.expert_model_parallel_size ==1:
                    checkpoint_name = get_checkpoint_name(args.save, 0, True, None, tp_rank, None, False)
                for k, v in full_model.items():
                    if not isinstance(v, torch.Tensor):
                        target_v = v
                    elif 'linear_qkv.weight' in k:
                        viewed = v.view(args.num_query_groups, -1, head_dim, args.hidden_size)
                        viewed = viewed[group_per_split*tp_rank : group_per_split*(tp_rank + 1)]
                        target_v = viewed.view(-1, args.hidden_size)
                    elif 'linear_qkv.bias' in k:
                        viewed = v.view(args.num_query_groups, -1, head_dim)
                        viewed = viewed[group_per_split*tp_rank : group_per_split*(tp_rank + 1)]
                        target_v = viewed.view(-1)
                    elif 'linear_proj' in k:
                        seg = v.shape[1] // args.tensor_model_parallel_size
                        target_v = v[:, seg*tp_rank : seg*(tp_rank + 1)]
                    elif 'embedding' in k or 'output_layer' in k:
                        seg = v.shape[0] // args.tensor_model_parallel_size
                        target_v = v[seg*tp_rank : seg*(tp_rank + 1)]
                    elif 'local_experts' in k:
                        expert_rank = int(re.findall(pattern, k)[0])
                        if expert_rank // num_local_experts != ep_rank:
                            continue
                        expert_local_rank = expert_rank % num_local_experts
                        if 'linear_fc1' in k and 'norm' not in k:
                            viewed = v.view(-1, args.moe_ffn_hidden_size, args.hidden_size)
                            seg = args.moe_ffn_hidden_size // args.tensor_model_parallel_size
                            target_v = viewed[:, seg*tp_rank: seg*(tp_rank+1), :].reshape(-1, args.hidden_size)
                        elif 'linear_fc2' in k:
                            seg = v.shape[1] // args.tensor_model_parallel_size
                            target_v = v[:, seg*tp_rank : seg*(tp_rank + 1)]
                        k = k.replace(f'local_experts.{expert_rank}', f'local_experts.{expert_local_rank}')
                    elif 'shared_expert' in k and 'gate' not in k:
                        if 'linear_fc1' in k:
                            viewed = v.view(-1, args.shared_moe_ffn_hidden_size, args.hidden_size)
                            seg = args.shared_moe_ffn_hidden_size // args.tensor_model_parallel_size
                            target_v = viewed[:, seg*tp_rank: seg*(tp_rank+1), :].reshape(-1, args.hidden_size)
                        elif 'linear_fc2' in k:
                            seg = v.shape[1] // args.tensor_model_parallel_size
                            target_v = v[:, seg*tp_rank : seg*(tp_rank + 1)]
                    else:
                        target_v = v
                    model_split[k] = target_v
                save_state_dict(args, [model_split], checkpoint_name)

    elif (
        args.pipeline_model_parallel_size > 1
        and args.num_experts is None
    ):

        for tp_rank in range(args.tensor_model_parallel_size):
            for pp_rank in range(args.pipeline_model_parallel_size):
                model_split = {}
                layer_offset = pp_rank * num_layers
                layers_to_copy = {}
                for layer in range(num_layers):
                    pp_layer_id = layer + layer_offset
                    layers_to_copy[f"decoder.layers.{pp_layer_id}"] = layer
                checkpoint_name = get_checkpoint_name(args.save, 0, True, True, tp_rank, pp_rank)
                print(f'tensor_parallel & pipeline_parallel, save model to {checkpoint_name}')
                for k, v in full_model.items():
                    if check_layer(layers_to_copy, k):
                        pattern = re.compile(r'\d+')
                        res = pattern.findall(k)
                        k = re.sub(r"decoder.layers.\d+", "decoder.layers." + str(layers_to_copy["decoder.layers." + res[0]]), k)
                    elif not ("word_embeddings" in k or "output_layer" in k or "final_layernorm" in k):
                        continue
                    if not isinstance(v, torch.Tensor):
                        target_v = v
                    elif 'linear_qkv.weight' in k:
                        viewed = v.view(args.num_query_groups, -1, head_dim, args.hidden_size)
                        viewed = viewed[group_per_split*tp_rank : group_per_split*(tp_rank + 1)]
                        target_v = viewed.view(-1, args.hidden_size)
                    elif 'linear_qkv.bias' in k:
                        viewed = v.view(args.num_query_groups, -1, head_dim)
                        viewed = viewed[group_per_split * tp_rank: group_per_split * (tp_rank + 1)]
                        target_v = viewed.view(-1)
                    elif 'linear_proj' in k or 'linear_fc2' in k:
                        seg = v.shape[1] // args.tensor_model_parallel_size
                        target_v = v[:, seg*tp_rank : seg*(tp_rank + 1)]
                    elif 'embedding' in k or 'output_layer' in k:
                        seg = v.shape[0] // args.tensor_model_parallel_size
                        target_v = v[seg*tp_rank : seg*(tp_rank + 1)]
                    elif 'linear_fc1' in k and 'norm' not in k:
                        viewed = v.view(-1, args.ffn_hidden_size, args.hidden_size)
                        seg = args.ffn_hidden_size // args.tensor_model_parallel_size
                        target_v = viewed[:, seg*tp_rank: seg*(tp_rank+1), :].reshape(-1, args.hidden_size)
                    else:
                        target_v = v
                    if "word_embeddings" in k:
                        if pp_rank == 0:
                            model_split[k] = target_v
                    elif "output_layer" in k or "final_layernorm" in k:
                        if pp_rank == args.pipeline_model_parallel_size - 1:
                            model_split[k] = target_v
                    else:
                        model_split[k] = target_v
                save_state_dict(args, [model_split], checkpoint_name)

    elif (
        args.pipeline_model_parallel_size > 1
        and args.num_experts % args.expert_model_parallel_size == 0
    ):

        for tp_rank in range(args.tensor_model_parallel_size):
            for ep_rank in range(args.expert_model_parallel_size):
                for pp_rank in range(args.pipeline_model_parallel_size):
                    model_split = {}
                    layer_offset = pp_rank * num_layers
                    layers_to_copy = {}
                    for layer in range(num_layers):
                        pp_layer_id = layer + layer_offset
                        layers_to_copy[f"decoder.layers.{pp_layer_id}"] = layer
                    if args.expert_model_parallel_size > 1:
                        checkpoint_name = get_checkpoint_name(args.save, 0, True, True, tp_rank, pp_rank, True, ep_rank)
                    elif args.expert_model_parallel_size == 1:
                        checkpoint_name = get_checkpoint_name(args.save, 0, True, True, tp_rank, pp_rank, False)
                    print(f'tensor_parallel & pipeline_parallel & expert_parallel, save model to {checkpoint_name}')
                    for k, v in full_model.items():
                        if check_layer(layers_to_copy, k):
                            layer_pattern = re.compile(r'\d+')
                            res = layer_pattern.findall(k)
                            k = re.sub(r"decoder.layers.\d+", "decoder.layers." + str(layers_to_copy["decoder.layers." + res[0]]), k)
                        elif not ("word_embeddings" in k or "output_layer" in k or "final_layernorm" in k):
                            continue
                        if not isinstance(v, torch.Tensor):
                            target_v = v
                        elif 'linear_qkv.weight' in k:
                            viewed = v.view(args.num_query_groups, -1, head_dim, args.hidden_size)
                            viewed = viewed[group_per_split * tp_rank: group_per_split * (tp_rank + 1)]
                            target_v = viewed.view(-1, args.hidden_size)
                        elif 'linear_qkv.bias' in k:
                            viewed = v.view(args.num_query_groups, -1, head_dim)
                            viewed = viewed[group_per_split * tp_rank: group_per_split * (tp_rank + 1)]
                            target_v = viewed.view(-1)
                        elif 'linear_proj' in k:
                            seg = v.shape[1] // args.tensor_model_parallel_size
                            target_v = v[:, seg * tp_rank: seg * (tp_rank + 1)]
                        elif 'embedding' in k or 'output_layer' in k:
                            seg = v.shape[0] // args.tensor_model_parallel_size
                            target_v = v[seg * tp_rank: seg * (tp_rank + 1)]
                        elif 'local_experts' in k:
                            expert_rank = int(re.findall(pattern, k)[0])
                            if expert_rank // num_local_experts != ep_rank:
                                continue
                            expert_local_rank = expert_rank % num_local_experts
                            if 'linear_fc1' in k and 'norm' not in k:
                                viewed = v.view(-1, args.moe_ffn_hidden_size, args.hidden_size)
                                seg = args.moe_ffn_hidden_size // args.tensor_model_parallel_size
                                target_v = viewed[:, seg * tp_rank: seg * (tp_rank + 1), :].reshape(-1,
                                                                                                    args.hidden_size)
                            elif 'linear_fc2' in k:
                                seg = v.shape[1] // args.tensor_model_parallel_size
                                target_v = v[:, seg * tp_rank: seg * (tp_rank + 1)]
                            k = k.replace(f'local_experts.{expert_rank}', f'local_experts.{expert_local_rank}')
                        elif 'shared_expert' in k and 'gate' not in k:
                            if 'linear_fc1' in k:
                                viewed = v.view(-1, args.shared_moe_ffn_hidden_size, args.hidden_size)
                                seg = args.shared_moe_ffn_hidden_size // args.tensor_model_parallel_size
                                target_v = viewed[:, seg * tp_rank: seg * (tp_rank + 1), :].reshape(-1,
                                                                                                    args.hidden_size)
                            elif 'linear_fc2' in k:
                                seg = v.shape[1] // args.tensor_model_parallel_size
                                target_v = v[:, seg * tp_rank: seg * (tp_rank + 1)]
                        else:
                            target_v = v
                        if "word_embeddings" in k:
                            if pp_rank == 0:
                                model_split[k] = target_v
                        elif "output_layer" in k or "final_layernorm" in k:
                            if pp_rank == args.pipeline_model_parallel_size - 1:
                                model_split[k] = target_v
                        else:
                            model_split[k] = target_v
                    save_state_dict(args, [model_split], checkpoint_name)

    else:
        raise ValueError('Something is wrong, please check your tp/pp/ep size')

    print(f'megatron model is save to {args.save}')

def check_hf_mg_forward(hfmodel, mgmodel, mgargs):
    hf_hiddens = [{} for _ in range(mgargs.num_layers)]
    mg_hiddens = [{} for _ in range(mgargs.num_layers)]

    hidden_size = mgargs.hidden_size
    vocab_size = mgargs.padded_vocab_size


    def print_input_hook(module, args, kwargs, layer_idx, mode):
        frame, name = mode.split('-')
        if frame == 'hf':
            hf_hiddens[layer_idx][name] = args[0].transpose(0, 1)
        elif frame == 'mg' and 'layer' in mode:
            mg_hiddens[layer_idx][name] = kwargs.get('hidden_states')
        elif frame == 'mg':
            mg_hiddens[layer_idx][name] = args[0]

    def print_output_hook(module, args, kwargs, output, layer_idx, mode):
        frame, name = mode.split('-')
        if mode in ['hf-lmhead']:
            hf_hiddens[layer_idx][name] = output.transpose(0, 1).reshape(-1, vocab_size)
            hf_hiddens[layer_idx][name + "_weight"] = module.weight
            hf_hiddens[layer_idx][name + '_token'] = output.transpose(0, 1).max(dim=-1)[1]
        elif mode in ['mg-lmhead']:
            mg_hiddens[layer_idx][name] = output[0].reshape(-1, vocab_size)
            mg_hiddens[layer_idx][name + "_weight"] = module.weight
            mg_hiddens[layer_idx][name + '_token'] = output[0].max(dim=-1)[1]
        elif mode in ['hf-o_proj_out']:
            hf_hiddens[layer_idx][name] = output
            hf_hiddens[layer_idx][name + '_weight'] = module.weight
        elif mode in ['mg-o_proj_out']:
            mg_hiddens[layer_idx][name] = output[0].reshape(-1, hidden_size)
            mg_hiddens[layer_idx][name + '_weight'] = module.weight
        elif mode in ['hf-attn_out']:
            hf_hiddens[layer_idx][name] = output[0].reshape(-1, hidden_size)
        elif mode in ['mg-attn_out']:
            mg_hiddens[layer_idx][name] = output[0].reshape(-1, hidden_size)

    if mgargs.untie_embeddings_and_output_weights:
        hfmodel.lm_head.register_forward_hook(partial(print_output_hook, layer_idx=mgargs.num_layers - 1, mode='hf-lmhead'),
                                            with_kwargs=True)

        mgmodel.output_layer.register_forward_hook(
            partial(print_output_hook, layer_idx=mgargs.num_layers - 1, mode='mg-lmhead'), with_kwargs=True)

    for idx, layer in enumerate(hfmodel.model.layers):

        layer.register_forward_pre_hook(partial(print_input_hook, layer_idx=idx, mode='hf-layer_in'), with_kwargs=True)

        layer.self_attn.o_proj.register_forward_pre_hook(partial(print_input_hook, layer_idx=idx, mode='hf-o_proj_in'),
                                                         with_kwargs=True)

        layer.self_attn.o_proj.register_forward_hook(partial(print_output_hook, layer_idx=idx, mode='hf-o_proj_out'),
                                                     with_kwargs=True)

        layer.self_attn.register_forward_hook(partial(print_output_hook, layer_idx=idx, mode='hf-attn_out'),
                                              with_kwargs=True)


    for idx, layer in enumerate(mgmodel.decoder.layers):

        layer.register_forward_pre_hook(partial(print_input_hook, layer_idx=idx, mode='mg-layer_in'), with_kwargs=True)

        layer.self_attention.linear_proj.register_forward_pre_hook(
            partial(print_input_hook, layer_idx=idx, mode='mg-o_proj_in'), with_kwargs=True)

        layer.self_attention.linear_proj.register_forward_hook(
            partial(print_output_hook, layer_idx=idx, mode='mg-o_proj_out'), with_kwargs=True)

        layer.self_attention.register_forward_hook(partial(print_output_hook, layer_idx=idx, mode='mg-attn_out'),
                                                   with_kwargs=True)


    input_ids = torch.tensor([[151644,   8506,  22564,  27608,  75188,   4344, 121395,  61991,  79554, 36689]]).long().cuda()
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(input_ids, -100, True, True, True)
    print(hfmodel)
    print(mgmodel)
    is_oom = False
    with torch.inference_mode():
        try:
            hfmodel.cuda()
            hflogits = hfmodel(input_ids=input_ids).logits
        except torch.cuda.OutOfMemoryError:
            print('oom for huggingface model forward')
            is_oom = True
        hfmodel.cpu()
        del hfmodel

    with torch.inference_mode():
        try:
            mgmodel.cuda()
            mglogits = mgmodel(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
        except torch.cuda.OutOfMemoryError:
            print('oom for megatron model forward')
            is_oom = True
        mgmodel.cpu()
        del mgmodel

    epsilon = 1e-5
    for idx, (hfh, mgh) in enumerate(zip(hf_hiddens, mg_hiddens)):
        assert len(hfh) == len(mgh)
        for k, hfv in hfh.items():
            mgv, hfv = mgh[k].cpu(), hfv.cpu()
            same_num = (hfv != mgv).sum()
            diff_num = ((hfv - mgv) > epsilon).sum()
            diff_max = (hfv - mgv).abs().max()
            print(f'layer:{idx}, {k}, diff: {same_num}, diff>{epsilon}:[{diff_num}/{hfv.numel()}] diff_max:{diff_max}')

    if not is_oom:
        same_num = (hflogits != mglogits).sum()
        diff_num = ((hflogits - mglogits) > epsilon).sum()
        diff_max = (hflogits - mglogits).abs().max()
        print(f'logits: {same_num}, diff>{epsilon}:[{diff_num}/{hflogits.numel()}] diff_max:{diff_max}')


def add_extra_args(parser):
    parser = get_patch_args(parser)
    parser = add_model_args(parser)
    return parser

def main():
    initialize_megatron(extra_args_provider=add_extra_args)
    args = get_args()

    if args.convert_checkpoint_from_megatron_to_transformers:
        config = AutoConfig.from_pretrained(args.hf_ckpt_path)
        mg_model = load_megatron_model(args)
        hf_model = AutoModelForCausalLM.from_pretrained(args.hf_ckpt_path, torch_dtype=config.torch_dtype)
        convert_checkpoint_from_megatron_to_transformers(mg_model, hf_model, args)
        save_hfmodel(args, hf_model)
    else:
        config = AutoConfig.from_pretrained(args.load)
        hf_model = AutoModelForCausalLM.from_pretrained(args.load, torch_dtype=config.torch_dtype)
        mg_model = model_provider()
        convert_checkpoint_from_transformers_to_megatron(hf_model, mg_model, args)
        if not args.num_experts:
            check_hf_mg_forward(hf_model, mg_model, args)
        save_mgmodel(mg_model, args)

if __name__ == "__main__":
    main()
