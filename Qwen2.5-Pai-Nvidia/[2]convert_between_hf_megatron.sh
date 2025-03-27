#!/bin/bash
export PYTHONWARNINGS="ignore::FutureWarning"
export PYTHONPATH=/workspace/Pai-Megatron-Patch-0.10.3:/workspace/Pai-Megatron-Patch-0.10.3/Megatron-LM-core_r0.11.1:$PYTHONPATH   


# --------------------------------------------------------
# paths
CONVERT_TYPE=megatron-to-huggingface    # 🚩

# >>>>>>> 以下是当 CONVERT_TYPE=huggingface-to-megatron 时的路径设置，否则注释掉以下
# LOAD_CHECKPOINT_PATH=/mnt/public/models/Qwen2.5-14B-Instruct      # 原始 huggingface模型路径
# SAVE_CHECKPOINT_PATH=/mnt/public/training_models/Qwen2.5-14B-Instruct  # 这个路径下必须是空白的

# >>>>>>> 以下是当 CONVERT_TYPE=megatron-to-huggingface 时的路径设置，否则注释掉以下
HF_MODEL_PATH=/mnt/public/models/Qwen2.5-14B-Instruct  # 原始 huggingface模型路径
LOAD_CHECKPOINT_PATH=/mnt/public/train_finished_models  # 训练完成的模型
SAVE_CHECKPOINT_PATH=/mnt/public/train_finished_models/final   # 注这个路径下必须是空白的
# --------------------------------------------------------

echo "CONVERT_TYPE: $CONVERT_TYPE " 
if [ $CONVERT_TYPE = "megatron-to-huggingface" ]; then
    convert_args=(
        --convert-checkpoint-from-megatron-to-transformers
        --hf-ckpt-path ${HF_MODEL_PATH}
    )
elif [ $CONVERT_TYPE = "huggingface-to-megatron" ]; then
    convert_args=()
else
    echo "[ERROR] CONVERT_TYPE must be \"huggingface-to-megatron\" or \"megatron-to-huggingface\"." 
    exit 1
fi


distributed_args=(
    --nproc_per_node 1
    --nnodes 1
    --node_rank 0
    --master_addr localhost
    --master_port 21111
)

# 所有参数量的model_config_args见文件下方
model_config_args=(
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

additional_args=(
    --load $LOAD_CHECKPOINT_PATH
    --save $SAVE_CHECKPOINT_PATH
    --target-tensor-model-parallel-size 2
    --target-pipeline-model-parallel-size 2
    #--target-num-layers-per-virtual-pipeline-stage 1
    --micro-batch-size 1  # 随便写
    --save-interval 1     # 随便写
    # --------------------------
    # 以下参数需要和训练时（[3]run_mcore_qwen.sh 中的 GPT_MODEL_ADDITIONAL_ARGS）相同
    --swiglu
    --seq-length 2048
    --no-async-tensor-model-parallel-allreduce
    --patch-tokenizer-type Qwen2Tokenizer
    --no-bias-swiglu-fusion
    --no-rope-fusion
    --use-rotary-position-embeddings
    --disable-bias-linear
    --add-qkv-bias
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --rotary-base 1000000
    # --------------------------
    --use-mcore-models
    --save-safetensors
    --bf16
    --transformer-impl transformer_engine  # 转换为 mcore 格式模型
)


torchrun ${distributed_args[@]} /workspace/Pai-Megatron-Patch-0.10.3/toolkits/model_checkpoints_convertor/qwen/hf2mcore_qwen2_dense_and_moe_gqa.py \
    ${model_config_args[@]} \
    ${additional_args[@]} \
    ${convert_args[@]} \

set +x

<<'EOF'



【附: qwen 系列 config】

if [ $MODEL_SIZE = 0.5B ]; then
    GPT_MODEL_CONFIGJSON_ARGS=(
        --num-layers 24
        --hidden-size 896 
        --num-attention-heads 14
        --ffn-hidden-size 4864    # 模型config.json里的INTERMEDIATE_SIZE
        --max-position-embeddings 32768
        --group-query-attention 
        --num-query-groups 2       # 模型config.json里的NUM_KEY_VALUE_HEADS
        --normalization RMSNorm 
        --norm-epsilon 1e-6    # 模型config.json里的RMS_NORM_EPS
        --extra-vocab-size 293  # 模型config.json里的EXTRA_VOCAB_SIZE
        # --untie-embeddings-and-output-weights # qwen2.5系列：如果参数量>=7B,则启用--untie-embeddings-and-output-weights，否则不启用
        # --use-cpu-initialization  # 只在72B模型时启用
    )

elif [ $MODEL_SIZE = 1.5B ]; then
    GPT_MODEL_CONFIGJSON_ARGS=(
        --num-layers 28
        --hidden-size 1536 
        --num-attention-heads 12
        --ffn-hidden-size 8960   
        --max-position-embeddings 32768
        --group-query-attention 
        --num-query-groups 2      
        --normalization RMSNorm 
        --norm-epsilon 1e-6   
        --extra-vocab-size 293 
        # --untie-embeddings-and-output-weights
        # --use-cpu-initialization 
    )

elif [ $MODEL_SIZE = 3B ]; then
    GPT_MODEL_CONFIGJSON_ARGS=(
        --num-layers 36
        --hidden-size 2048 
        --num-attention-heads 16
        --ffn-hidden-size 11008   
        --max-position-embeddings 32768
        --group-query-attention 
        --num-query-groups 2       
        --normalization RMSNorm 
        --norm-epsilon 1e-6   
        --extra-vocab-size 293
        # --untie-embeddings-and-output-weights 
        # --use-cpu-initialization  
    )

elif [ $MODEL_SIZE = 7B ]; then
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

elif [ $MODEL_SIZE = 14B ]; then
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

elif [ $MODEL_SIZE = 32B ]; then
    GPT_MODEL_CONFIGJSON_ARGS=(
        --num-layers 64
        --hidden-size 5120 
        --num-attention-heads 40
        --ffn-hidden-size 27648   
        --max-position-embeddings 131072
        --group-query-attention 
        --num-query-groups 8      
        --normalization RMSNorm 
        --norm-epsilon 1e-5   
        --extra-vocab-size 421 
        --untie-embeddings-and-output-weights 
        # --use-cpu-initialization  
    )

elif [ $MODEL_SIZE = 72B ]; then
    GPT_MODEL_CONFIGJSON_ARGS=(
        --num-layers 80
        --hidden-size 8192 
        --num-attention-heads 64 
        --ffn-hidden-size 29568   
        --max-position-embeddings 131072
        --group-query-attention 
        --num-query-groups 8       
        --normalization RMSNorm 
        --norm-epsilon 1e-5   
        --extra-vocab-size 421 
        --untie-embeddings-and-output-weights 
        --use-cpu-initialization  
    )
fi

EOF