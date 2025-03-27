## METAX Megatron-LM ChatGLM3:

# 1. 使用Megatron-LM镜像
Megatron-LM在/workspace文件夹下

# 2. 设置环境变量&perf
cd /workspace/Megatron-LM/examples
source env.sh  
bash perf.sh performance  


# 3. 数据和tokenizer准备：
准备Megatron数据如：oscar-en-10k-meg-llama_text_document.bin 和oscar-en-10k-meg-llama_text_document.idx  
准备chatglm3的模型文件夹  

# 4. hf模型文件转Megatron
bash convert_chatglm3_dp8.sh <原始模型文件夹> <模型转换之后的模型文件夹>

# 5. 单机8卡pretrain 6B (加载模型）
5.1 修改pretrain_chatglm3-6B_8gpu.sh：  
MODEL_PATH：转换后的模型路径  
DATA_PATH：数据路径  
5.2 运行模型：  
bash pretrain_chatglm3-6B_8gpu.sh  