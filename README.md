<p align="center">
    <h1 align="center" style="margin-bottom:0px;">&nbsp;Qwen2.5-Pai&nbsp;</h1>
    <h6 align="center">在无问芯穹上使用 Pai-Megatron 进行 qwen2.5 训练</h6>
</p>

## 🪶版本

### Qwen2.5-Pai-Nvidia

- Megatron-Core：0.11.0+ （2025.3.23 main 分支）
- Pai-Megatron-Patch：0.10.3
- cuda 12.4

### Qwen2.5-Pai-MetaX
- Megatron-Core：未知版本（maca 2.27镜像里的）
- Pai-Megatron-Patch：未知版本（maca 2.27镜像里的）
- maca 2.27 

## 🪶说明

### Qwen2.5-Pai-Nvidia

- 将 Pai 中对 qwen 2.5 进行训练的部分尽可能地适配 Megatron-Core 0.11.0

### Qwen2.5-Pai-MetaX

- 修改了模型权重转换脚本为正确的层级格式；修复数据预处理、日志、保存检查点等逻辑，对沐曦中 Megatron-core 的核心功能未做修改


## 安装

### 英伟达

使用镜像：infini-ai/ngc:pytorch-24.03-py3 或更高，需要确保 cuda 至少 12.4（无问芯穹预制镜像里优先选 ngc）

按照 Megatron-LM 官方仓库指引，安装 apex 和 TransformerEngine（https://github.com/NVIDIA/TransformerEngine）

注1： 本机的 gcc 和 g++ 版本不能太低，不然无法编译，见 https://github.com/google/tensorstore/issues/63 （若 ubuntu 小于 22.04 ，需要升级 gcc）

注2： Megatron-LM 本身不需要编译，只是其依赖的 apex 库需要编译，Megatron-LM 会从 apex 库里调用.so 文件

最后将这个仓库的 [Nvidia]Pai-Megatron-Patch 移动到镜像中 /workspace 目录下

### 沐曦

使用镜像：cr.infini-ai.com/te-c7vnqlzlmvzffc2v/megatron-lm:maca2.27.0.7-py310

在镜像中有一个沐曦自带的 /workspace/Pai-Megatron-Patch，其中有一个指向沐曦版 Megatron-LM 的软连接，<del>你会发现按沐曦的 readme 运行不起来</del>，忽略它们或把它们移动到别的地方；使用这个仓库的 [MetaX-maca2.27]Pai-Megatron-Patch 替换镜像中原本的 /workspace/Pai-Megatron-Patch 文件夹

最后需要确认 transformers >=4.47.0，否则无法保存模型为 safetensors


## 1. 数据预处理

原始 jsonl 文件中不需要有任何 chat template 或特殊 token（pad, eos 等）

### pretrain

原始数据文件需要是 jsonl 格式，其每条样本需要具有一个"text"字段，示例如下：

    {"text": "自然语言处理是人工智能的一个重要领域。\n它使机器能够理解并生成自然语言文本。"}

运行 [1]preprocess_pretrain.sh 对数据进行处理，会在 `--output-prefix` 路径下生成 bin 和 idx 文件

### sft

原始数据文件需要是 jsonl 格式，其每条样本需要具有"instruction", "input", "output"字段，示例如下：

    {"instruction": "This is system prompt...","input": "This is user prompt...","output": "This is assistant output..."},

运行 [1]preprocess_sft.sh 对数据进行处理，会在 `--output-prefix` 路径下生成 bin 和 idx 文件

## 2. 转换 huggingface 为 Megatron 格式

⚠️ 目前沐曦 maca2.27 不支持将 huggingface 模型转为 distcp，因此沐曦脚本中转换后的模型仍为 torch 格式

在 [2]convert_between_hf_megatron.sh 中，指定 `CONVERT_TYPE` 为 `huggingface-to-megatron`（训练前 huggingface 转 megatron），然后逐个参数对比校验，tp、pp 需要与之后训练时一致，模型参数需要与实际模型的 config 一致


## 3. 训练

运行 [3]run_mcore_qwen.sh 进行训练，按照注释修改参数

**注：该文件为单节点配置，未添加多节点相关环境变量**

在**首次**训练时，需要确认 `LOAD_CHECKPOINT_PATH` 目录结构如下:

    LOAD_CHECKPOINT_PATH  
    ├── release
    │   ├── mp_rank_00_000
    │   │   └── model_optim_rng.pt
    │   ├── mp_rank_00_001    
    │   └── ...
    ├── config.json  
    ├── tokenizer.json  
    ├── ... 
    └── latest_checkpointed_iteration.txt 

训练过程中的检查点会保存在 `SAVE_TORCH_CHECKPOINT_PATH` 下:

    SAVE_TORCH_CHECKPOINT_PATH  
    ├── iter_0000100  
    ├── iter_0000200  
    ├── ... 
    └── latest_checkpointed_iteration.txt 

在**从上次暂停处继续训练**时，需要指定 `LOAD_CHECKPOINT_PATH` 与 `SAVE_TORCH_CHECKPOINT_PATH` 相同

## 4. 转换 Megatron 为 huggingface 格式

### 英伟达

在 [3]run_mcore_qwen.sh 中，指定 `TRAIN_OR_CONVERT_CKPT` 为 `convert`，其它参数和训练时保持不变

在转换前，需要确认 `LOAD_CHECKPOINT_PATH` 目录结构如下:

    LOAD_CHECKPOINT_PATH  
    ├── iter_0000200  
    │   ├── _0_0.distcp
    │   ├── ...
    │   └── common.pt
    ├── config.json  
    ├── tokenizer.json  
    ├── ... 
    └── latest_checkpointed_iteration.txt 

运行脚本，会从 `LOAD_CHECKPOINT_PATH` 中查找最新的分布式检查点，转为常规 torch 格式并保存到 `SAVE_TORCH_CHECKPOINT_PATH`，得到文件目录如下：

    SAVE_TORCH_CHECKPOINT_PATH  
    ├── iter_0000200  
    │   ├── mp_rank_00_000
    │   │   ├── model_optim_rng.pt
    │   │   └── distrib_optim.pt
    │   ├── mp_rank_00_001
    │   └── ...
    └── latest_checkpointed_iteration.txt 


然后在 [2]convert_between_hf_megatron.sh 中，指定 `CONVERT_TYPE` 为 `megatron-to-huggingface`（训练完成后 megatron 转 huggingface），指定 `LOAD_CHECKPOINT_PATH` 为刚才 [3]run_mcore_qwen.sh 中的 SAVE_TORCH_CHECKPOINT_PATH ；指定本脚本中 `SAVE_CHECKPOINT_PATH` 为保存最终 huggingface 格式模型的路径。

其它参数和步骤 2 中填写时保持不变即可，运行脚本，得到文件目录如下：

    SAVE_CHECKPOINT_PATH  
    ├── config.json
    ├── tokenizer.json
    ├── model-00001-of-00003.safetensors
    ├── model-00002-of-00003.safetensors
    └── ...

### 沐曦

在 [2]convert_between_hf_megatron.sh 中，指定 `CONVERT_TYPE` 为 `megatron-to-huggingface`（训练完成后 megatron 转 huggingface），指定本脚本中 `SAVE_CHECKPOINT_PATH` 为保存最终 huggingface 格式模型的路径。

在转换前，需要确认 `LOAD_CHECKPOINT_PATH` 目录结构如下:

    LOAD_CHECKPOINT_PATH  
    ├── iter_0000200  
    │   ├── mp_rank_00_000
    │   │   ├── model_optim_rng.pt
    │   │   └── distrib_optim.pt
    │   ├── mp_rank_00_001
    │   └── ...
    └── latest_checkpointed_iteration.txt 

其它参数和步骤 2 中填写时保持不变即可，运行脚本，得到文件目录如下：

    SAVE_CHECKPOINT_PATH  
    ├── config.json
    ├── tokenizer.json
    ├── model-00001-of-00003.safetensors
    ├── model-00002-of-00003.safetensors
    └── ...
