## 沐曦Qwen 使用说明

# 1. 使用Megatron-LM镜像
workspace中的Pai-Megatron-Path为工程目录  

# 3. 设置环境变量&perf
cd how-to-run
bash perf.sh performance

# 4. 数据和tokenizer准备：
准备Megatron数据如：oscar-en-10k-meg-llama_text_document.bin 和oscar-en-10k-meg-llama_text_document.idx  
准备Qwen-14B的模型文件  

# 5. hf到Megatron模型文件转换
5.1 cd toolkits/model_checkpoints_convertor/qwen  
修改convert_qwen_14B_pp4.sh中第三行和第四行为：下载的模型文件和转换后的模型文件  
5.2 运行转换脚本
模型转tp1pp4
bash convert_qwen_14B_pp4.sh  

# 6. 双机16卡pretrain 14B
6.1 修改配置  
cd /software/kanliu/firmwork/Pai-Megatron-Patch/examples/qwen  
修改pretrain_Qwen-14B-2k_16gpu.sh或者pretrain_Qwen-14B-4k_16gpu.sh：  
DATASET_PATH:数据路径   
PRETRAIN_CHECKPOINT_PATH:转换之后的模型路径   
OUTPUT_BASEPATH:输出路径   

6.2 两台机器分别运行脚本  
bash pretrain_Qwen-14B-2k_16gpu.sh 2 0 <主节点ip>  
bash pretrain_Qwen-14B-2k_16gpu.sh 2 1 <主节点ip>  

或者：  
bash pretrain_Qwen-14B-4k_16gpu.sh 2 0 <主节点ip>  
bash pretrain_Qwen-14B-4k_16gpu.sh 2 1 <主节点ip>
