#!/bin/bash
set -e  
ENV=$1
export PYTHONWARNINGS="ignore::FutureWarning"
export SSL_CERT_FILE=/home/cacert.pem
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FLASH_ATTN=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#export UB_SKIPMC=1  
export MCCL_MAX_NCHANNELS=16
export PYTHONPATH=/workspace/Pai-Megatron-Patch:/workspace/Pai-Megatron-Patch/Megatron-LM:$PYTHONPATH  

# --------------------------------------------------------
IS_INITIAL_RUN=true    
# --------------------------------------------------------
DATASET_PATH=/mnt/public/data/spitzblattr/demo/demo_text_document  
LOAD_CHECKPOINT_PATH=/mnt/public/model/spitzblattr/training_models/Qwen2.5-7B-Instruct              
SAVE_TORCH_CHECKPOINT_PATH=/mnt/public/model/spitzblattr/training_models/saved_ckpts 
WANDB_SAVE_PATH=/home/wandb_logs   
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
    --train-mode finetune 
    --transformer-impl local  
)
num_epoches=3 
num_training_samples=154678    
train_iters=$(( num_epoches * num_training_samples / ${TRAINING_ARGS[4]} ))    
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
    )
fi


EVAL_LOGGING_AND_SAVE_ARGS=(
    --save-interval 100 
    --eval-iters 50      
    --eval-interval 100  
    --save $SAVE_TORCH_CHECKPOINT_PATH 
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
EVAL_LOGGING_AND_SAVE_ARGS+=(
    --wandb-project spec_decode   
    --wandb-exp-name qwen-sft  
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