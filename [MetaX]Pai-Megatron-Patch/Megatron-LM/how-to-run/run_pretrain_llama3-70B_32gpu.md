## METAX Megatron-LM LLAMA3-70B训练流程：

# 1. 使用Megatron-LM镜像
Megatron-LM在/workspace文件夹下

# 2. 设置环境变量&perf
cd /workspace/Megatron-LM/examples
source env.sh  
bash perf.sh performance

# 3. 数据和tokenizer准备：
准备Megatron数据如：oscar-en-10k-meg-llama_text_document.bin 和oscar-en-10k-meg-llama_text_document.idx  
准备meta的llama3模型文件

# 4. 四机32卡pretrain 70B (不加载模型）
4.1 pretrain_llama3_70B_meg_32gpu.sh中DATA_DIR和TDIR为数据和llama3-70B模型路径  
四台机器分别运行：   
bash pretrain_llama3_70B_meg_32gpu.sh 4 0 <主节点ip>  
bash pretrain_llama3_70B_meg_32gpu.sh 4 1 <主节点ip>  
bash pretrain_llama3_70B_meg_32gpu.sh 4 2 <主节点ip>  
bash pretrain_llama3_70B_meg_32gpu.sh 4 3 <主节点ip>  