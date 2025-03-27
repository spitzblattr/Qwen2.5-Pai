## 使用说明

# 1. 使用Megatron-LM镜像
workspace中的Pai-Megatron-Path为工程目录  
将workspace中的Megatron-LM拷贝到Pai-Megatron-Path文件夹中  （需要Megatron-LM-core-0.8.0版本）

# 2. 安装依赖：
其它：transformers  
apt-get install build-essential

# 3. 设置环境变量&perf
cd examples  
source env.sh  
bash perf.sh performance

# 4. 数据和tokenizer准备：
准备Megatron数据如：mmap_deepseekv2_datasets_text_document.bin 和mmap_deepseekv2_datasets_text_document.idx   
下载DeepSeek-Coder-V2-Lite-Base的模型  


# 5. hf到megatron模型转换
5.1 cd toolkits/model_checkpoints_convertor/deepseek  
修改SOURCE_CKPT_PATH 和 TARGET_CKPT_PATH 分别为下载的模型文件和保存的模型文件  
5.2 运行转换脚本（目前仅支持tp1pp1ep8的模型转换）  
bash convert_tp1pp1ep8.sh      

# 6. 双机16卡训练
6.1 修改训练脚本  
cd examples/deepseek_v2   
修改run_mcore_deepseek_code_lite_base.sh：    
DATASET_PATH:数据路径   
VALID_DATASET_PATH: valid数据路径    
PRETRAIN_CHECKPOINT_PATH:转换之后的模型路径   
OUTPUT_BASEPATH:输出路径   
   

6.2 运行脚本 
两台机器分别运行：    
bash run_mcore_deepseek_code_lite_base.sh 2 0 <master node ip>   
bash run_mcore_deepseek_code_lite_base.sh 2 1 <master node ip>    



# 7. megatron模型转hf   
cd toolkits/model_checkpoints_convertor/deepseek  
修改convert_tp1pp1ep8.sh  ：   
MG2HF=true   
HF_CKPT_PATH为原始huggingface下载文件夹   
bash convert_tp1pp1ep8.sh 