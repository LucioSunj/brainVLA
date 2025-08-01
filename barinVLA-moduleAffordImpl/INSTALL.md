# Hume模型安装和设置指南

本指南将帮助您安装Hume模型并在LIBERO基准测试上运行它。

## 前提条件

- Python 3.8+（主要包推荐使用Python 3.10）
- CUDA支持（LIBERO评估需要CUDA 11.3+）
- [uv包管理器](https://docs.astral.sh/uv/getting-started/installation/)

## 安装步骤

### 1. 安装uv包管理器

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. 克隆仓库并初始化子模块

```bash
git clone https://github.com/hume-vla/hume.git
cd hume
git submodule update --init --recursive
```

### 3. 设置主环境

```bash
# 使用uv同步依赖
GIT_LFS_SKIP_SMUDGE=1 uv sync
```

### 4. 设置LIBERO环境（用于评估）

```bash
# 为LIBERO创建单独的Python 3.8虚拟环境
uv venv --python 3.8 experiments/libero/.venv

# 激活LIBERO环境
source experiments/libero/.venv/bin/activate

# 安装LIBERO依赖
uv pip sync experiments/libero/requirements.txt 3rd/LIBERO/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e 3rd/LIBERO
```

## 下载模型检查点

根据您想要评估的LIBERO任务套件，下载相应的检查点：

| 任务套件 | 检查点链接 |
|------------|----------------|
| LIBERO-GOAL | [hume-libero-goal](https://huggingface.co/Hume-vla/Libero-Goal-2) |
| LIBERO-OBJECT | [hume-libero-object](https://huggingface.co/Hume-vla/Libero-Object-1) |
| LIBERO-SPATIAL | [hume-libero-spatial](https://huggingface.co/Hume-vla/Libero-Spatial-1) |

下载这些检查点并将它们放置在适当的目录中（例如，`exported/libero/`）。

## 环境配置

创建或修改`scripts/env.sh`文件，填入您的具体路径：

```bash
# 设置您的LEROBOT_DATASET路径
export HF_LEROBOT_HOME="/path/to/your/LEROBOT_DATASET"
# 设置您的TRITON_CACHE_DIR路径
export TRITON_CACHE_DIR="/path/to/your/TRITON_CACHE"
export TOKENIZERS_PARALLELISM=false

# WANDB配置（可选）
export WANDB_API_KEY="your_wandb_api_key"
export WANDB_PROJECT=Hume
# 如果需要，将WANDB设置为离线模式
export WANDB_MODE=offline
export WANDB_ENTITY="your_wandb_entity"
```

## 运行LIBERO评估

1. 更新`experiments/libero/scripts/eval_libero.sh`中的检查点路径，指向您下载的检查点
2. 运行评估脚本：

```bash
source experiments/libero/.venv/bin/activate  # 确保您在LIBERO环境中
bash experiments/libero/scripts/eval_libero.sh
```

### 测试时采样（TTS）参数

评估性能可以通过不同的TTS参数进行调整。以下是一些推荐配置：

```bash
# 配置1
s2_candidates_num=5
noise_temp_lower_bound=1.0
noise_temp_upper_bound=1.0
time_temp_lower_bound=0.9
time_temp_upper_bound=1.0

# 配置2
s2_candidates_num=5
noise_temp_lower_bound=1.0
noise_temp_upper_bound=1.2
time_temp_lower_bound=1.0
time_temp_upper_bound=1.0

# 配置3
s2_candidates_num=5
noise_temp_lower_bound=1.0
noise_temp_upper_bound=2.0
time_temp_lower_bound=1.0
time_temp_upper_bound=1.0
```

您可以在`eval_libero.sh`脚本中修改这些参数，以优化不同任务的性能。

## 故障排除

- 如果遇到与CUDA相关的错误，请确保您安装了正确版本的CUDA（LIBERO需要CUDA 11.3）
- 对于依赖问题，请检查所有必需的包是否已安装在正确的环境中
- 如果策略服务器启动失败，请检查端口可用性 