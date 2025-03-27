## 沐曦Qwen 使用说明

# 1. 使用Megatron-LM镜像
workspace中的Pai-Megatron-Path为工程目录  

# 3. 设置环境变量&perf
cd how-to-run
bash perf.sh performance

# 4. 数据和tokenizer准备：
准备Megatron数据如：oscar-en-10k-meg-llama_text_document.bin 和oscar-en-10k-meg-llama_text_document.idx  
准备Qwen1.5-72B的模型文件  

# 5. hf到Megatron模型文件转换
Pai-megatron中，Qwen1.5不支持pp>1的转换

# 6. 4机32卡pretrain 14B
6.1 修改配置  
cd /workspace/Pai-Megatron-Patch/examples/qwen  
修改pretrain_Qwen1.5-72B_32gpu.sh:  
DATASET_PATH:数据路径   
PRETRAIN_CHECKPOINT_PATH:模型路径   
OUTPUT_BASEPATH:输出路径   

6.2 两台机器分别运行脚本  
bash pretrain_Qwen1.5-72B_32gpu.sh 4 0 <主节点ip>  
bash pretrain_Qwen1.5-72B_32gpu.sh 4 1 <主节点ip>  
bash pretrain_Qwen1.5-72B_32gpu.sh 4 2 <主节点ip>  
bash pretrain_Qwen1.5-72B_32gpu.sh 4 3 <主节点ip>  
