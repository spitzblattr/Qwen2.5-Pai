## METAX Megatron-LM ChatGLM2:

# 1. 使用Megatron-LM镜像
Megatron-LM在/workspace文件夹下

# 2. 设置环境变量&perf
cd /workspace/Megatron-LM/examples
source env.sh  
bash perf.sh performance


# 3. 数据和tokenizer准备：
准备Megatron数据如：oscar-en-10k-meg-llama_text_document.bin 和oscar-en-10k-meg-llama_text_document.idx  
准备chatglm2的tokenizer的文件：tokenizer.model

# 4. 单机8卡pretrain 6B (不加载模型）
4.1 修改pretrain_chatglm2-6B_8gpu.sh中DATA_PATH和TOKENIZER_PATH为数据和tokenizer文件路径  
4.2 训练：bash pretrain_chatglm2-6B_8gpu.sh