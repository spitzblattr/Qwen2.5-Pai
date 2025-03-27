## METAX Megatron-LM LLAMA2-70B：

# 1. 使用Megatron-LM镜像  
Megatron-LM在/workspace文件夹下  

# 2. 设置环境变量&perf  
cd /workspace/Megatron-LM/examples  
source env.sh  
bash perf.sh performance  

# 3. 数据和tokenizer准备：
准备Megatron数据如：oscar-en-10k-meg-llama_text_document.bin 和oscar-en-10k-meg-llama_text_document.idx  
准备llama2-70B的tokenizer的文件：tokenizer.model  

# 4.  8机64卡pretrain 70B 64GPU（不加载模型）  

在8个节点上分别运行:   
bash pretrain_llama2_70B_64gpu.sh 8 0 <主节点ip>    
bash pretrain_llama2_70B_64gpu.sh 8 1 <主节点ip>    
bash pretrain_llama2_70B_64gpu.sh 8 2 <主节点ip>     
bash pretrain_llama2_70B_64gpu.sh 8 3 <主节点ip>    
bash pretrain_llama2_70B_64gpu.sh 8 4 <主节点ip>    
bash pretrain_llama2_70B_64gpu.sh 8 5 <主节点ip>    
bash pretrain_llama2_70B_64gpu.sh 8 6 <主节点ip>   
bash pretrain_llama2_70B_64gpu.sh 8 7 <主节点ip>   