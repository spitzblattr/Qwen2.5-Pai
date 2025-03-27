#!/bin/bash
set -ex

sh run_pretrain_qwen_72B_tp8pp4_32gpu.sh  \
dsw  \
72B   \
1    \
1024 \
1e-5   \
1e-6   \
4096  \
4096  \
bf16  \
8  \
4  \
1 \
none  \
true   \
true  \
true   \
false   \
100000  \
/pde_ai/datasets/llama3_data/wudao_llama3bpe_content_document \
/pde_ai/models/llm/Qwen/Qwen2-72B-Instruct/ \
100000000   \
10000   \
output_TP8-PP4 \
${1} \
${2} \
${3} 
