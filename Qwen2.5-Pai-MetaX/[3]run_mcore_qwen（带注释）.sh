#!/bin/bash
set -e  
ENV=$1
export PYTHONWARNINGS="ignore::FutureWarning"
export SSL_CERT_FILE=/home/cacert.pem
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FLASH_ATTN=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#export UB_SKIPMC=1     # ← 如果有AssertionError: CUDA device, driver and/or toolkit version does not support comm+GEMM overlap with CUDA Multicast. Launch app with UB_SKIPMC=1 to try CUDA IPC instead
export MCCL_MAX_NCHANNELS=16
export PYTHONPATH=/workspace/Pai-Megatron-Patch:/workspace/Pai-Megatron-Patch/Megatron-LM:$PYTHONPATH  

# --------------------------------------------------------
IS_INITIAL_RUN=true    # 🚩 # 是否为首次训练
# 💡沐曦上由于训练过程中保存的检查的始终是torch格式而不是torch-dist，所以训练完毕时不需要再次运行这个脚本从torch-dist转回torch，直接可以运行[2]convert_between_hf_megatron.sh
# --------------------------------------------------------
DATASET_PATH=/mnt/public/data/spitzblattr/demo/demo_text_document  # 这个路径后面会直接追加".bin"/"idx"
LOAD_CHECKPOINT_PATH=/mnt/public/model/spitzblattr/training_models/Qwen2.5-7B-Instruct                    
# ↑↑ LOAD_CHECKPOINT_PATH 下必须要有config.json、vocab.json、tokenizer.json等，并且还要有一个latest_checkpointed_iteration.txt
SAVE_TORCH_CHECKPOINT_PATH=/mnt/public/model/spitzblattr/training_models/saved_ckpts 
WANDB_SAVE_PATH=/home/wandb_logs   # Path to save the wandb results locally.
# --------------------------------------------------------
# ⭐️ ⭐️ ⭐️：该section下所有参数都必须校验
# ⭐️：该单个参数必须校验（如果没有这个标记，并不意味着这个参数一定不需要校验，只是它*或许可以*不校验）
# 🌙：这个参数是 PAI 新增的，不是基础 Megatron 里的
# 所有参数见 Pai-Megatron-Patch/Megatron-LM/megatron/training/arguments.py

DATA_ARGS=(
    --data-path $DATASET_PATH  # 关于混合比例数据集(blended dataset)的解释：https://github.com/NVIDIA/Megatron-LM/issues/547
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
# 7B
GPT_MODEL_CONFIGJSON_ARGS=(
    --num-layers 28
    --hidden-size 3584 
    --num-attention-heads 28
    --ffn-hidden-size 18944   
    --max-position-embeddings 131072
    --group-query-attention 
    --num-query-groups 4      
    --normalization RMSNorm 
    --norm-epsilon 1e-6    
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
    --max-position-embeddings 32768    # qwen2.5
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
    # --calculate-per-token-loss    # 沐曦版Megatron未实现
    --train-mode finetune  # ⭐️🌙  # type=str, help="pretrain or finetune"。*必须指定该参数* # 该参数影响 megatron_patch/template/helper.py 中的 get_batch 函数
    --transformer-impl local  # 沐曦2.27不支持 transformer_engine
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
    #--optimizer hybridadam    # 🌙 # 沐曦2.27未实现 # 相关代码：https://github.com/alibaba/Pai-Megatron-Patch/pull/298/files (只存在于PAI-Megatron-LM-240718分支)
    #--optimizer-offload-policy auto     # 🌙 # 沐曦2.27未实现
)
# if --tensor-model-parallel-size > 1
if [ ${MODEL_PARALLEL_ARGS[1]} -gt 1 ]; then
    MODEL_PARALLEL_ARGS+=(
        --sequence-parallel  # When --sequence-parallel is used, sequence_len must be a multiple of --tensor-parallel. 
        --context-parallel-size 1    # Degree of context parallelism. 默认为1.
        # 以下 沐曦2.27未实现，需要 transformer engine 和 mcore 格式模型
        #--tp-comm-overlap  
        #--overlap-grad-reduce
        #--overlap-param-gather   
    )
fi
# if --pipeline-model-parallel-size > 1
if [ ${MODEL_PARALLEL_ARGS[3]} -gt 1 ]; then
    MODEL_PARALLEL_ARGS+=(
        # Enable p2p comm overlap when PP > 1 by setting num_layers_per_virtual_pipeline_stage.
        # --num-layers-per-virtual-pipeline-stage 1
        # 沐曦未实现
    )
fi


EVAL_LOGGING_AND_SAVE_ARGS=(
    --save-interval 100 # ⭐️
    --eval-iters 50      # Number of iterations to run for evaluation validation/test for. default=100
    --eval-interval 100  # ⭐️ # Interval between running evaluation on validation set. default=1000
    --save $SAVE_TORCH_CHECKPOINT_PATH 
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
EVAL_LOGGING_AND_SAVE_ARGS+=(
    #--async-save   # 沐曦 Megatron未实现
    # 不需要指定 wandb 的 team(shsfcx)，这会在第一次运行脚本时手动让你输入
    --wandb-project spec_decode   # ⭐️
    --wandb-exp-name qwen-sft  # ⭐️ 
    --wandb-save-dir $WANDB_SAVE_PATH 
)



torchrun ${DISTRIBUTED_ARGS[@]} /workspace/Pai-Megatron-Patch/examples/qwen2_5/pretrain_qwen.py \
    ${DATA_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${GPT_MODEL_CONFIGJSON_ARGS[@]} \
    ${GPT_MODEL_ADDITIONAL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${EVAL_LOGGING_AND_SAVE_ARGS[@]} \

set +x


<<'EOF'


EOF