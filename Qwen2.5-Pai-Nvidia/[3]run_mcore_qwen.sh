#!/bin/bash
set -e  
ENV=$1
export PYTHONWARNINGS="ignore::FutureWarning"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FLASH_ATTN=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#export UB_SKIPMC=1    
export PYTHONPATH=/workspace/Pai-Megatron-Patch-0.10.3:/workspace/Pai-Megatron-Patch-0.10.3/Megatron-LM-core_r0.11.1:$PYTHONPATH   


# --------------------------------------------------------
IS_INITIAL_RUN=false   
TRAIN_OR_CONVERT_CKPT=convert   
# --------------------------------------------------------
DATASET_PATH=/mnt/public/datasets/max_len_2048/max_len_2048_text_document  
LOAD_CHECKPOINT_PATH=/mnt/public/training_models/saved_ckpts                     
SAVE_DIST_CHECKPOINT_PATH=/mnt/public/training_models/saved_ckpts
WANDB_SAVE_PATH=/home/wandb_logs   
SAVE_TORCH_CHECKPOINT_PATH=/mnt/public/train_finished_models
# --------------------------------------------------------

DATA_ARGS=(
    --data-path $DATASET_PATH  
    --split 98,2,0
    --seq-length 2048   
    --dataset MMAP  
)

DISTRIBUTED_ARGS=(
    --nproc_per_node 8
    --nnodes 1
    --node_rank 0 
    --master_addr localhost
    --master_port 22222
)

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

GPT_MODEL_ADDITIONAL_ARGS=(
    --swiglu  
    --position-embedding-type rope  
    --disable-bias-linear 
    --add-qkv-bias   
    --rotary-base 1000000    
    --max-position-embeddings 32768    
    --rotary-percent 1.0   
    --rotary-seq-len-interpolation-factor 1 
    --patch-tokenizer-type Qwen2Tokenizer 
)


TRAINING_ARGS=(
    --bf16  
    --micro-batch-size 2 
    --global-batch-size 16 
    --weight-decay 0.01  
    --init-method-std 0.02
    --clip-grad 1.0 
    --lr 6.5e-5       
    --min-lr 2.5e-6   
    --lr-warmup-iters 0
    --lr-decay-style cosine 
    --attention-dropout 0.0    
    --hidden-dropout 0.0   
    --calculate-per-token-loss   
    --train-mode finetune  
)
num_epoches=3  
num_training_samples=154678    
train_iters=$(( num_epoches * num_training_samples / ${TRAINING_ARGS[4]} ))    # TRAINING_ARGS[4]是--global-batch-size
TRAINING_ARGS+=(
    --train-iters $train_iters   
)


MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size 2  
	--pipeline-model-parallel-size 2 
    --recompute-activations 
    --use-distributed-optimizer  
)
# if --tensor-model-parallel-size > 1
if [ ${MODEL_PARALLEL_ARGS[1]} -gt 1 ]; then
    MODEL_PARALLEL_ARGS+=(
        --sequence-parallel  
        --context-parallel-size 1  
        --tp-comm-overlap 
        --overlap-grad-reduce
        --overlap-param-gather
    )
fi

EVAL_LOGGING_AND_SAVE_ARGS=(
    --save-interval 100 
    --eval-iters 50   
    --eval-interval 100  
    --save $SAVE_DIST_CHECKPOINT_PATH 
    --load $LOAD_CHECKPOINT_PATH 
    --exit-on-missing-checkpoint  
    --auto-detect-ckpt-format 
    --log-interval 100    
    --log-throughput  
)
if [ ${IS_INITIAL_RUN} = true ]; then
    EVAL_LOGGING_AND_SAVE_ARGS+=(
        --no-load-optim 
        --no-load-rng 
    )
fi
if [ $TRAIN_OR_CONVERT_CKPT = "train" ]; then
    EVAL_LOGGING_AND_SAVE_ARGS+=(
        --async-save
        --wandb-project spec_decode  
        --wandb-exp-name qwen-sft
        --wandb-save-dir $WANDB_SAVE_PATH 
    )
elif [ $TRAIN_OR_CONVERT_CKPT = "convert" ]; then
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


EOF