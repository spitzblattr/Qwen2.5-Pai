## METAX Megatron-LM LLAMA2：

# 1. 使用Megatron-LM镜像
Megatron-LM在/workspace文件夹下

# 2. 设置环境变量&perf
cd /workspace/Megatron-LM/examples
source env.sh  
bash perf.sh performance


# 3. 数据和tokenizer准备：
准备Megatron数据如：oscar-en-10k-meg-llama_text_document.bin 和oscar-en-10k-meg-llama_text_document.idx    
准备llama2-7B的tokenizer的文件：tokenizer.model  

# 4. 单机8卡pretrain 7B (不加载模型）
4.1 修改pretrain_llama2_7B_8gpu.sh中DATA_PATH和TOKENIZER_PATH为数据和tokenizer文件路径  
4.2 训练：bash pretrain_llama2_7B_8gpu.sh  
4.3 更优性能推荐使用qadam，将/pde_ai/models/llm/qadam.cpython-38-x86_64-linux-gnu.so拷贝至Megatron-LM/megatron文件夹下即可
