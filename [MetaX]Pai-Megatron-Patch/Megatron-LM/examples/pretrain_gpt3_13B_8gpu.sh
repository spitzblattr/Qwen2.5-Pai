#!/bin/bash
set -x
# Runs the "13B" parameter model
export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

DROP_OUT=0.0

#CHECKPOINT_PATH="/external/ckpt/gpt3_5b" #<Specify path>
TENSORBOARD_LOGS_PATH="/external/ckpt/tensorboard" #<Specify path>
VOCAB_FILE="/external/models/gpt2-tokenizer/gpt2-vocab.json" #<Specify path to file>/gpt2-vocab.json
MERGE_FILE="/external/models/gpt2-tokenizer/gpt2-merges.txt" #<Specify path to file>/gpt2-merges.txt
DATA_PATH="/external/datasets/RedPajama-Data-1T-Sample/gpt-merged/gpt-text-merge_seq-len2048" #<Specify path and file prefix>_text_document


DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

GPT_MODEL_ARGS=(
    --num-layers 40 
    --hidden-size 5120 
    --num-attention-heads 40
    --seq-length 2048
    --max-position-embeddings 2048
)

TRAINING_ARGS=(
    --micro-batch-size 2
    --global-batch-size 1024
    --train-iters 500 
    --attention-dropout ${DROP_OUT} \
    --hidden-dropout ${DROP_OUT} \
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    --bf16
    --lr 6.0e-5 
    --lr-decay-style cosine 
    --min-lr 6.0e-6
    --lr-warmup-fraction .001 
    --lr-decay-iters 430000 
    --use-mcore-models
    --use-distributed-optimizer
    --overlap-grad-reduce
    --use-flash-attn
    --transformer-impl local
)

MODEL_PARALLEL_ARGS=( 
	--tensor-model-parallel-size 1
	--pipeline-model-parallel-size 4
    --pipline-num-layers-list 9 10 10 11	
)

DATA_ARGS=(
    --data-path $DATA_PATH 
    --vocab-file $VOCAB_FILE 
    --merge-file $MERGE_FILE 
    --split 949,50,1
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --save-interval 10000 
    --eval-interval 1000 
#    --save $CHECKPOINT_PATH 
#    --load $CHECKPOINT_PATH 
    --eval-iters 100
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
)

torchrun ${DISTRIBUTED_ARGS[@]} ../pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}
