#!/bin/bash
export PYTHONPATH=/workspace/Pai-Megatron-Patch-0.10.3:/workspace/Pai-Megatron-Patch-0.10.3/Megatron-LM-core_r0.11.1:$PYTHONPATH  

# --input: 若--partitions大于1，这里需要是一个含有多个jsonl文件的文件夹，否则需要是单个jsonl文件的路径
# --seq-length：对每个样本 pad 或截断到这个长度
# --output-prefix: 这个路径后面会直接追加".bin"/".idx"作为输出文件
# --load：path to load tokenizer config file.
# --workers：A good default for fast pre-processing is: (workers * partitions) = available CPU cores.
# --partitions：分区处理文件的数量（共生成几个.bin / 几个.idx 文件）

# 可选：--sequence-packing
# 可选：--keep-sequential-samples，action='store_true'. Ensure ordering of samples in jsonl files is preserved when using partitions>1.


python /workspace/Pai-Megatron-Patch-0.10.3/toolkits/sft_data_preprocessing/build_idxmap_sft_dataset.py \
    --input /mnt/public/datasets/all_train_data_0930_153478.jsonl \
    --patch-tokenizer-type Qwen2Tokenizer \
    --seq-length 111 \
    --output-prefix /mnt/public/datasets/max_len_111/max_len_111 \
    --load /mnt/public/models/Qwen2.5-0.5B-Instruct \
    --workers 8 \
    --partitions 1

