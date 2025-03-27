## 沐曦 Megatron-LM GPT3：

# 1. 使用Megatron-LM镜像
Megatron-LM在/workspace文件夹下

# 2. 设置环境变量&perf
cd /workspace/Megatron-LM/examples
source env.sh  
bash perf.sh performance

# 3. 数据和tokenizer准备：
准备Megatron数据如：oscar-en-10k-meg-llama_text_document.bin 和oscar-en-10k-meg-llama_text_document.idx    
准备gpt3-6.7B的tokenizer的文件：gpt2-vocab.json 和 gpt2-merges.txt


# 4. 单机8卡pretrain 6.7B (不加载模型）
4.1 修改pretrain_gpt3_6.7B_8gpus.sh中DATA_PATH，VOCAB_FILE和MERGE_FILE为相应的路径
4.2 训练：bash pretrain_gpt3_6.7B_8gpus.sh