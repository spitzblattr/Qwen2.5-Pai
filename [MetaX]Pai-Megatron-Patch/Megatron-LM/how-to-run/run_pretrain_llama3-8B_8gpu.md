## METAX Megatron-LM LLAMA3-8B训练流程：

# 1. 使用Megatron-LM镜像
Megatron-LM在/workspace文件夹下

# 2. 设置环境变量&perf
cd /workspace/Megatron-LM/examples
source env.sh  
bash perf.sh performance

# 3. 数据和tokenizer准备：
准备Megatron数据如：oscar-en-10k-meg-llama_text_document.bin 和oscar-en-10k-meg-llama_text_document.idx  
准备meta的llama3模型文件

# 4. 单机8卡pretrain 8B (不加载模型）
4.1 pretrain_llama3_8B_meg_8gpu.sh中DATA_DIR和TDIR为数据和llama3路径  
4.2 训练：bash pretrain_llama3_8B_meg_8gpu.sh