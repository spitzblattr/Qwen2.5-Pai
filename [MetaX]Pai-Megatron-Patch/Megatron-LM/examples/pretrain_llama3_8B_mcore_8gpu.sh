#! /bin/bash

# Setting the environment variables
export MACA_PATH=/opt/maca
export MACA_CLANG_PATH=${MACA_PATH}/mxgpu_llvm/bin
export MACA_CLANG=${MACA_PATH}/mxgpu_llvm
export DEVINFO_ROOT=${MACA_PATH}
export CUCC_PATH=${MACA_PATH}/tools/cu-bridge
export CUDA_PATH=${CUCC_PATH}
export PATH=${CUCC_PATH}:${MACA_PATH}/bin:${MACA_CLANG}/bin:${PATH}
export LD_LIBRARY_PATH=${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${LD_LIBRARY_PATH}
export CUDA_DEVICE_MAX_CONNECTIONS=1
#export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:64"
export MACA_SMALL_PAGESIZE_ENABLE=1

export MCPYTORCH_DISABLE_PRINT=1

export MCCL_NET_GDR_LEVEL=7
export MCCL_P2P_LEVEL=SYS
export MCCL_LIMIT_RING_LL_THREADTHRESHOLDS=1
export FORCE_ACTIVATE_WAIT=1

export SET_DEVICE_NUMA_PREFERRED=1


export MCCL_MAX_NCHANNELS=16
#export MHA_USE_BLAS=ON
#export MCBLAS_CUSTOMIZED_CONFIG_PATH=/workspace/Megatron-LM/examples/mcblas_customized_config.yaml
export MHA_BWD_NO_ATOMIC_F64=1
export SET_DEVICE_NUMA_PREFERRED=1

export MAX_JOBS=20
export PYTORCH_ENABLE_SAME_RAND_A100=1

# Runs the "LLaMA3-8B" parameter model

DATA_DIR="/external/datasets/llama3_data/wudao_llama3bpe_content_document"
GPUS_PER_NODE=8
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=22543
MBS=1
ITERS=100
TP=1
PP=2
TDIR="/external/models/llm/Llama/Meta-Llama-3-8B" #-Instruct"
MEGAPATH="../"
export PYTHONPATH=$PYTHONPATH:$MEGAPATH
# algorithm args

MODEL_ARGS=" \
    --num-layers 32 \
    --hidden-size 4096 \
    --ffn-hidden-size 14336 \
    --num-attention-heads 32 \
    --group-query-attention \
    --num-query-groups 8 \
    --seq-length 8192 \
    --max-position-embeddings 8192 \
    --swiglu \
    --normalization RMSNorm \
    --norm-epsilon 1.0e-5 \
    --global-batch-size 512 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --attention-softmax-in-fp32"

OPT_ARGS=" \
    --lr 1.0e-5 \
    --train-iters $ITERS \
    --lr-decay-iters $ITERS \
    --lr-decay-style cosine \
    --min-lr 1.0e-6 \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.008"
    
ALGO_ARGS="$MODEL_ARGS $OPT_ARGS"

# data args

DATA_ARGS=" \
    --data-path $DATA_DIR \
    --tokenizer-type Llama3Tokenizer \
    --tokenizer-model $TDIR \
    --split 100,0,0
"

# training args

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

TRAINING_ARGS=" \
    --micro-batch-size $MBS \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --pipline-num-layers-list 16 16 \
    --sequence-parallel \
    --bf16 \
    --no-repeat-kv \
    --accumulate-bf16 1
"

RECOMPUTE_ARGS="
       --recompute-granularity full \
       --recompute-method block \
       --recompute-num-layers 1 \
       --recompute-num-layers-list 5 0 \
"
# vendor args

VENDOR_ARGS=" \
    --transformer-impl local \
    --use-distributed-optimizer \
    --use-mcore-models \
    --use-flash-attn
"

BASE_PATH=./llama3-8B-save
LOG_NAME=llama3-8b_pretrain_106_${WORLD_SIZE}_TP${TP}_PP${PP}
LOG_PATH=${BASE_PATH}/log/${LOG_NAME}/node${NODE_RANK}.log
mkdir -p ${BASE_PATH}/log/${LOG_NAME}




OUTPUT_ARGS=" --log-interval 1"

run_cmd="torchrun $DISTRIBUTED_ARGS $MEGAPATH/pretrain_gpt.py \
    $ALGO_ARGS \
    $DATA_ARGS \
    $TRAINING_ARGS \
    $VENDOR_ARGS \
    $RECOMPUTE_ARGS \
    $OUTPUT_ARGS \
    "

echo ${run_cmd}
${CMD} 2>&1 | tee ${LOG_PATH}
eval ${run_cmd}


