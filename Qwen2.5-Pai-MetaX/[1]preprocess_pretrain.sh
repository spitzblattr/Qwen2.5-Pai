#!/bin/bash
export PYTHONPATH=/workspace/Pai-Megatron-Patch:/workspace/Pai-Megatron-Patch/Megatron-LM:$PYTHONPATH  

# --input: 若--partitions大于1，这里需要是一个含有多个jsonl文件的文件夹，否则需要是单个jsonl文件的路径
# --seq-length：对每个样本 pad 或截断到这个长度
# --output-prefix: 这个路径后面会直接追加".bin"/".idx"作为输出文件
# --load：path to load tokenizer config file.
# --workers：A good default for fast pre-processing is: (workers * partitions) = available CPU cores.
# --partitions：分区处理文件的数量

# 可选：--sequence-packing
# 可选：--keep-sequential-samples，action='store_true'. Ensure ordering of samples in jsonl files is preserved when using partitions>1.


python /workspace/Pai-Megatron-Patch/toolkits/pretrain_data_preprocessing/preprocess_data_megatron.py \
    --input /mnt/public/data/spitzblattr/demo \
    --json-keys text \
    --patch-tokenizer-type Qwen2Tokenizer \
    --seq-length 2048 \
    --output-prefix /mnt/public/data/spitzblattr/demo/demo \
    --load /mnt/public/model/huggingface/Qwen2.5-0.5B-Instruct \
    --keep-sequential-samples \
    --append-eod \
    --workers 8 \
    --partitions 2

