## 沐曦Qwen 使用说明

# 1. 使用Megatron-LM镜像
workspace中的Pai-Megatron-Path为工程目录  
将workspace中的Megatron-LM拷贝到Pai-Megatron-Path文件夹中

# 2. 安装依赖：
曦适配whl包：flash attn  
其它：transformers  
apt-get install build-essential

# 3. 设置环境变量&perf
cd examples  
source env.sh  
bash perf.sh performance

# 4. 数据和tokenizer准备：
准备Megatron数据如：oscar-en-10k-meg-llama_text_document.bin 和oscar-en-10k-meg-llama_text_document.idx  
下载Qwen2的模型文件


# 5. hf到megatron模型转换
5.1 cd toolkits/model_checkpoints_convertor/qwen  
修改convert_qwen2_7B_pp2.sh中第三行和第四行为：  
下载的模型文件和保存的模型文件  
5.2 运行转换脚本  
bash convert_qwen2_7B_pp2.sh  

# 6. 单机8卡pretrain 7B
6.1 修改训练脚本  
cd /software/kanliu/firmwork/Pai-Megatron-Patch/examples/qwen2  
修改pretrain_Qwen2-7B_8gpu.sh：  
DATASET_PATH:数据路径   
PRETRAIN_CHECKPOINT_PATH:转换之后的模型路径   
OUTPUT_BASEPATH:输出路径   

6.2 运行脚本  
bash pretrain_Qwen2-7B_8gpu.sh  