#!/bin/bash
set -e  
ENV=$1
export PYTHONWARNINGS="ignore::FutureWarning"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FLASH_ATTN=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#export UB_SKIPMC=1      # 如果有 AssertionError: CUDA device, driver and/or toolkit version does not support comm+GEMM overlap with CUDA Multicast. Launch app with UB_SKIPMC=1 to try CUDA IPC instead.
export PYTHONPATH=/workspace/Pai-Megatron-Patch-0.10.3:/workspace/Pai-Megatron-Patch-0.10.3/Megatron-LM-core_r0.11.1:$PYTHONPATH   


# --------------------------------------------------------
IS_INITIAL_RUN=false    # 🚩 # 是否为首次训练
TRAIN_OR_CONVERT_CKPT=convert    # 🚩 # “train” | “convert”（不会训练，只是转换检查点）
# 在 convert 前，LOAD_CHECKPOINT_PATH 下需要有config.json、vocab.json、tokenizer.json等，并且还要有一个latest_checkpointed_iteration.txt
# --------------------------------------------------------
DATASET_PATH=/mnt/public/datasets/max_len_2048/max_len_2048_text_document  # 这个路径后面会直接追加".bin"/"idx"
LOAD_CHECKPOINT_PATH=/mnt/public/training_models/saved_ckpts                     
# ↑↑ LOAD_CHECKPOINT_PATH 下必须要有config.json、vocab.json、tokenizer.json等，并且还要有一个latest_checkpointed_iteration.txt
SAVE_DIST_CHECKPOINT_PATH=/mnt/public/training_models/saved_ckpts
WANDB_SAVE_PATH=/home/wandb_logs   # Path to save the wandb results locally.
SAVE_TORCH_CHECKPOINT_PATH=/mnt/public/train_finished_models
# --------------------------------------------------------
# ⭐️ ⭐️ ⭐️：该section下所有参数都必须校验
# ⭐️：该单个参数必须校验（如果没有这个标记，并不意味着这个参数一定不需要校验，只是它*或许可以*不校验）
# 🌙：这个参数是 PAI 新增的，不是基础 Megatron 里的
# 所有参数见 Pai-Megatron-Patch/Megatron-LM/megatron/training/arguments.py

DATA_ARGS=(
    --data-path $DATASET_PATH   # 关于混合比例数据集(blended dataset)的解释：https://github.com/NVIDIA/Megatron-LM/issues/547
    --split 98,2,0 # ⭐️ # 若要指定单独的valid数据集来源，使用--valid-data-path
    --seq-length 2048   # ⭐️ # Megatron-LM requires the input sequence length to be fixed and padded to the --seq-length
    --dataset MMAP  # 🌙  # "JSON-SFT"(未处理的.json文件) or "MMAP"(处理完毕的bin和idx)
)


# ⭐️ ⭐️ ⭐️
DISTRIBUTED_ARGS=(
    --nproc_per_node 8
    --nnodes 1
    --node_rank 0 
    --master_addr localhost
    --master_port 22222
)

# ⭐️ ⭐️ ⭐️
GPT_MODEL_CONFIGJSON_ARGS=(
    --num-layers 48
    --hidden-size 5120 
    --num-attention-heads 40
    --ffn-hidden-size 13824   
    --max-position-embeddings 131072
    --group-query-attention 
    --num-query-groups 8      
    --normalization RMSNorm 
    --norm-epsilon 1e-5    
    --extra-vocab-size 421 
    --untie-embeddings-and-output-weights 
    # --use-cpu-initialization 
)
# ⭐️ ⭐️ ⭐️
GPT_MODEL_ADDITIONAL_ARGS=(
    --swiglu  
    --position-embedding-type rope  
    --disable-bias-linear # 先去除所有线性层的 bias
    --add-qkv-bias   # 再在 q,k,v_proj 层添加 bias
    --rotary-base 1000000    # Base to use for rotary positional embeddings. 默认为10000
    --max-position-embeddings 32768    
    --rotary-percent 1.0   
    --rotary-seq-len-interpolation-factor 1 
    --patch-tokenizer-type Qwen2Tokenizer # 🌙 
)


TRAINING_ARGS=(
    --bf16  # ⭐️  
    --micro-batch-size 2 # ⭐️ 
    --global-batch-size 16 # ⭐️ # should be a multiple of micro-batch-size times data-parallel-size。当显式指定了 --global-batch-size 时，Megatron 会根据此值自动推断 num_micro_batches 的大小
    --weight-decay 0.01  
    --init-method-std 0.02
    --clip-grad 1.0 
    --lr 6.5e-5       # ⭐️
    --min-lr 2.5e-6   # ⭐️
    --lr-warmup-iters 0
    --lr-decay-style cosine 
    #--lr-decay-iters: If None defaults to `--train-iters`
    --attention-dropout 0.0     # 默认启用并为0.1
    --hidden-dropout 0.0    # 默认启用并为0.1
    --calculate-per-token-loss    # 🤔
    --train-mode finetune  # ⭐️🌙  # type=str, help="pretrain or finetune" 该参数影响 megatron_patch/template/helper.py 中的 get_batch 函数
)
num_epoches=3   # ⭐️
num_training_samples=154678    # ⭐️ # 自行查看
train_iters=$(( num_epoches * num_training_samples / ${TRAINING_ARGS[4]} ))    # TRAINING_ARGS[4]是--global-batch-size
TRAINING_ARGS+=(
    --train-iters $train_iters   
)


MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size 2   # ⭐️
	--pipeline-model-parallel-size 2  # ⭐️
    --recompute-activations  # 在多数情况下都应该启用，除非训练内存非常有限
    --use-distributed-optimizer   # 是否使用Megatron版Zero-1优化器: , action='store_true'
    # 2025.3.24: Megatron-LM 正在添加 pai 的 hybridadam
    # 相关代码及如何启用 cpu offload：https://github.com/NVIDIA/Megatron-LM/commit/60007c93d3aad8ce5fcca8c60267220e22f35b45#diff-f4b11f68d40efd8e51059784591dd551fe305f3640f09a99f11de599c6a58a79
    # --optimizer hybridadam    # 🌙 相关代码：https://github.com/alibaba/Pai-Megatron-Patch/pull/298/files (只存在于PAI-Megatron-LM-240718分支)
    # --optimizer-offload-policy auto     # 🌙
)
# if --tensor-model-parallel-size > 1
if [ ${MODEL_PARALLEL_ARGS[1]} -gt 1 ]; then
    MODEL_PARALLEL_ARGS+=(
        # 关于 sequence-parallel 和 context-parallel 的介绍：https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/context_parallel.html
        --sequence-parallel  # When --sequence-parallel is used, sequence_len must be a multiple of --tensor-parallel.
        --context-parallel-size 1    # Degree of context parallelism. 默认为1.
        --tp-comm-overlap  # 如果TP>1，则启用overlap相关参数
        --overlap-grad-reduce
        --overlap-param-gather
    )
fi
# if --pipeline-model-parallel-size > 1
if [ ${MODEL_PARALLEL_ARGS[3]} -gt 1 ]; then
    MODEL_PARALLEL_ARGS+=(
        # Enable p2p comm overlap when PP > 1 by setting num_layers_per_virtual_pipeline_stage.
        # --num-layers-per-virtual-pipeline-stage 1
        # ⬆要启用这个参数，需要修改 Pai-Megatron-Patch-0.10.3/toolkits/model_checkpoints_convertor/qwen/hf2mcore_qwen2_dense_and_moe_gqa.py 里的相关函数
        # 尝试参考 pai 仓库里的 hf2mcore_qwen2.5_vl.py 改了一下，失败（模型各层划分看起来没问题但loss增大），找不出原因，暂时搁置一下
    )
fi


EVAL_LOGGING_AND_SAVE_ARGS=(
    --save-interval 100 # ⭐️
    --eval-iters 50    # Number of iterations to run for evaluation validation/test for. default=100
    --eval-interval 100  # ⭐️ # Interval between running evaluation on validation set. default=1000
    --save $SAVE_DIST_CHECKPOINT_PATH 
    --load $LOAD_CHECKPOINT_PATH 
    --exit-on-missing-checkpoint    # 如果找不到--load中的路径，直接退出
    --auto-detect-ckpt-format   # 每次（重新或继续）训练前，自动检查$LOAD_CHECKPOINT_PATH 的 checkpoint 是原始torch格式还是distcp格式
    
    --log-interval 100    
    --log-throughput   # If set, calculate and log throughput per GPU.
)
if [ ${IS_INITIAL_RUN} = true ]; then
    EVAL_LOGGING_AND_SAVE_ARGS+=(
        --no-load-optim   # When start initial training, specify `--no-load-optim --finetune` to make sure the optimizer state is NOT loaded from the pretrained GPT model so the continued pretraining is from a clean start. But, after the first run, you should launch without`--no-load-optim --finetune` to make sure the optimizer state is correctly loaded from your last job.
        --no-load-rng 
    )
fi
if [ $TRAIN_OR_CONVERT_CKPT = "train" ]; then
    EVAL_LOGGING_AND_SAVE_ARGS+=(
        --async-save   # Apply async checkpointing save. Currently works only with `torch_dist` distributed checkpoint format.
        # 不需要指定 wandb 的 team(shsfcx)，这会在第一次运行脚本时手动让你输入
        --wandb-project spec_decode   # ⭐️
        --wandb-exp-name qwen-sft  # ⭐️ 
        --wandb-save-dir $WANDB_SAVE_PATH 
    )
elif [ $TRAIN_OR_CONVERT_CKPT = "convert" ]; then
    # 参考 https://github.com/NVIDIA/Megatron-LM/issues/1266
    echo "正在将 $LOAD_CHECKPOINT_PATH 中的最新检查点从 torch_dist 转换到 torch 格式，并保存至 $SAVE_TORCH_CHECKPOINT_PATH" 
    EVAL_LOGGING_AND_SAVE_ARGS+=(
        --ckpt-convert-format torch  
        --ckpt-convert-save $SAVE_TORCH_CHECKPOINT_PATH
    )
fi


torchrun ${DISTRIBUTED_ARGS[@]} /workspace/Pai-Megatron-Patch-0.10.3/examples/qwen2/pretrain_qwen.py \
    ${DATA_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${GPT_MODEL_CONFIGJSON_ARGS[@]} \
    ${GPT_MODEL_ADDITIONAL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${EVAL_LOGGING_AND_SAVE_ARGS[@]} \

set +x


<<'EOF'

【？】
自从Megatron-core 0.11.0, 保存检查点的时候就会出现
[WARNING  | megatron.core.dist_checkpointing.validation]: There is difference in the common state dict in different ranks. The differences are {6: ([('optimizer', 'optimizer', 'param_groups', 1, 'step')], [], []), 7: ([('optimizer', 'optimizer', 'param_groups', 1, 'step')], [], [])}
[WARNING  | megatron.core.dist_checkpointing.validation]: There is difference in the common state dict in different ranks. The differences are {6: ([], [], [(('optimizer', 'optimizer', 'param_groups', 1, 'step'), <class 'int'>, <class 'int'>)]), 7: ([], [], [(('optimizer', 'optimizer', 'param_groups', 1, 'step'), <class 'int'>, <class 'int'>)])}
似乎是optimizer里多了一个step字段引起的
不影响训练和保存/加载检查点



EOF