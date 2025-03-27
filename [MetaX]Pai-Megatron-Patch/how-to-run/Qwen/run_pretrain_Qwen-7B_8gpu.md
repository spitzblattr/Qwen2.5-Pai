## 沐曦Qwen 使用说明

# 1. 使用Megatron-LM镜像
workspace中的Pai-Megatron-Path为工程目录  

# 3. 设置环境变量&perf
cd how-to-run
bash perf.sh performance

# 4. 数据和tokenizer准备：
准备Megatron数据如：oscar-en-10k-meg-llama_text_document.bin 和oscar-en-10k-meg-llama_text_document.idx  
准备Qwen-7B的模型文件  

# 5. hf到Megatron模型文件转换
5.1 cd toolkits/model_checkpoints_convertor/qwen  
修改convert_qwen_7B_pp2.sh中第三行和第四行分别为：  
下载的模型文件和转换后的模型文件  
5.2 运行转换脚本  
模型转tp1pp2： 
bash convert_qwen_7B_pp2.sh  

# 6. 单机8卡pretrain 7B
6.1 修改配置  
cd /software/kanliu/firmwork/Pai-Megatron-Patch/examples/qwen    
修改pretrain_Qwen-7B-4k_8gpu.sh或者pretrain_Qwen-7B-8k_8gpu.sh：  
DATASET_PATH:数据路径   
PRETRAIN_CHECKPOINT_PATH:转换之后的模型路径   
OUTPUT_BASEPATH:输出路径   

6.2 运行脚本  
bash pretrain_Qwen-7B-4k_8gpu.sh  
或者
bash pretrain_Qwen-7B-8k_8gpu.sh
