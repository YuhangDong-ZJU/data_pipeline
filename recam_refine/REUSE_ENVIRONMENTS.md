# 复用机器上已有的环境

已有环境可以直接用于处理，不必运行 `bootstrap.py`。复用模式先查找、导入和测试已有依赖，
没有完整可用的环境时，只向选中的已有环境补装缺失的包及其依赖，不升级、降级或重装已有包。
它支持 Python 3.10 / 3.11；原锁定 runtime 的用法继续保留。

## 根据已有安装脚本判断缺项

下面是安装配置能保证的范围；其他任务或手动安装可能已经补齐一些包，实际运行时会再检查一次。

| 已有环境 | 仓库安装配置 | refine 还需检查的缺项 |
| --- | --- | --- |
| `recam_data_pipeline` | `droid_depth/run_trim_padded_tails.sh` 显式安装 Python 3.10、NumPy、PyArrow、FFmpeg | Pillow、PyAV（`av`）、SciPy、Hugging Face Hub，以及几何/校验依赖 |
| `droid_normals` | `droid_normals/install_normalcrafter.sh` 使用固定 NormalCrafter requirements，含 NumPy、Pillow、Hub、NetworkX、Matplotlib、Torch | requirements 未显式列出 PyArrow、PyAV、SciPy、trimesh、pycollada、zstandard |
| `recam_download` | `dataset_download/download_recam_lerobot.sh` 只显式安装 Python、pip、Hub、hf_xet | 大部分媒体、Parquet、几何和 Torch 依赖；通常不应作为首选 |

NormalCrafter 配置来源：[固定提交的 requirements.txt](https://github.com/Binyr/NormalCrafter/blob/75af9887a2cb14cd1ce3883c5773bc296565777c/requirements.txt)。
原 H100 安装脚本默认 cu128，可通过变量覆盖；本任务以用户确认的已安装 **2.8.0+cu129** 为准，
不会重新运行 NormalCrafter 安装脚本或根据其默认值替换 Torch。

2026-09-08 只读检查示例服务器 codex-218：其 `droid_normals` 已有 PyArrow 25.0.0、PyAV 16.0.1、SciPy 1.15.3，
但缺 trimesh、pycollada、zstandard。它的 Torch 是 2.0.1，不能用于本任务 GPU 优化。
这不是对方 H100 机器的安装清单，不能将两台机器的实际状态混为一谈。

每次新开 CPU 或 GPU 终端，设置原来的数据、工作目录变量，再加：

```bash
export RECAM_REFINE_REUSE_ENV=1
```

随后 `run_step.sh` 的每一步都使用复用模式。也可以不设置这个变量，在每条命令末尾加 `--reuse-env`。
`--prepared-runtime` 可以保留，但不再触发旧版锁定环境的安装或版本核对。
在复用模式中，是否补装由新的环境检查控制；该旧参数不禁止补装缺项。

## 自动查找规则

Conda 位置按 `DATA_PIPELINE_CONDA_BIN`、`DROID_NORMALS_CONDA_BIN`、`DROID_DEPTH_CONDA_BIN`、
`MINIFORGE_HOME/bin/conda`、PATH、家目录下的 miniforge3/miniconda3 查找。
已有的 `TRIM_ENV_NAME`、`DROID_NORMALS_ENV_NAME`、`RECAM_DOWNLOAD_ENV_NAME` 覆盖设置继续生效。

CPU 优先测试 `recam_data_pipeline`，然后 `droid_normals`、`recam_download`；GPU 优先测试 `droid_normals`。
之后尝试当前工作目录已准备的 `runtime/env/bin/python`、激活的 Conda 环境及 PATH 中的 Python。
每个候选都会实际验证；名称相同不等于依赖已经满足。先尝试所有完整可用的环境，再选择缺项较少的环境补装，
优先保留已有合适 Torch，避免为了少量几何包另外下载整套 Torch。一次只补装一个环境；失败时不轮流修改其他环境。

若要指定已有环境，二选一；指定后验证失败会停止，不偷偷换环境：

```bash
export RECAM_REFINE_ENV_NAME=droid_normals
```

```bash
export RECAM_REFINE_PYTHON=/absolute/path/to/existing/env/bin/python
```

也可以按命令指定 `--conda-env droid_normals` 或 `--python /absolute/path/to/env/bin/python`。
不要同时设置两种选择方式。两个 GPU worker 可以共用同一个 Python 路径，无需复制环境。

## CPU：开始处理前检查已有环境

进入已更新的代码仓库后，执行下面的环境准备。已有包通过检查即复用，缺包则自动补装。
不修改数据集；报告、下载缓存和安装计划写入 RECAM_WORK。CPU 节点无需 GPU 或 NVIDIA 驱动。

```bash
# 迁移、解包、对齐等基础能力
python3 -m recam_refine.environment "$RECAM_WORK" --profile base

# 最终 CPU 校验能力；允许复用 2.8.0+cu129，在 CPU 上运算
python3 -m recam_refine.environment "$RECAM_WORK" --profile cpu

# 提前检查将供 GPU 使用的环境；此时只执行 CPU 运算，不探测 GPU
python3 -m recam_refine.environment "$RECAM_WORK" --profile prepare-gpu
```

这里代替原流程的两条 `bootstrap.py --cpu-torch` / `--prepare-gpu` 安装命令。
如果要在安装前先查看缺项，给上述命令加 `--check-only`。它会检查并报告，绝不安装；全部通过则返回 0，缺项则非 0。
正常流程不必额外执行这一轮，补装前已经会自动输出缺项和实际安装计划。
报告中的 `python` 是实际选中的解释器。GPU 节点需能访问该已有环境；如各节点 PATH/Conda 配置不同，
在两台 GPU 节点用 `RECAM_REFINE_PYTHON` 指向 CPU 检查通过的同一路径。

基础检查包括依赖导入、16 位 PNG 读写、Parquet 读写以及 H.264 编解码。
几何步骤还检查机器人网格采样。CPU 校验要求 PyTorch 2.8.0+cpu 或 2.8.0+cu129；
GPU 预检查和优化保持本次约定的 2.8.0+cu129，GPU 优化启动时还验证实际 CUDA 运算和批量优化器。
不会因为换环境降低原来的外参质量门槛。

只有缺失的直接依赖使用本仓库经过测试的版本；已安装的兼容版本照常保留。
补装前冻结所有已安装包的版本，用 pip dry-run 解析完整依赖，并拒绝任何替换旧包的计划。
先下载和校验全部 wheel，再使用本地 SHA-256 清单和 `--no-deps` 安装，之后验证旧版本不变、
没有新增依赖冲突、PNG/Parquet/H.264/几何能力可用。没有 pip 或 pip 太旧时，从工作目录运行固定版本安装工具，
不升级环境中的 pip。只安装二进制 wheel，不要求 gcc、nvcc 或 sudo。

已有包版本冲突、Torch 版本不符、系统 Python、只读环境或缺失系统动态库不会被盲目覆盖；输出原因并停止。
已有其他程序造成的 pip check 问题保留在前后报告中，不以自动升级旧包的方式修复它们。
网络失败可以重试相同命令；依赖完整时不访问安装源。
保留 shell 中的代理、CA、PIP_INDEX_URL/PIP_EXTRA_INDEX_URL/PIP_FIND_LINKS 等网络设置，
安装时忽略 pip 配置文件中的用户目录/其他目标路径等选项，以确保补装进入选定环境。

## CPU 和 GPU 执行命令

沿用原来逐步执行的阶段顺序。以第一步为例：

```bash
export RECAM_REFINE_REUSE_ENV=1
bash recam_refine/run_step.sh transfer "$RECAM_ROOT" "$RECAM_WORK" \
  --depth-output "$DEPTH_OUTPUT" --depth-chunks 2-13
```

GPU A：

```bash
export RECAM_REFINE_REUSE_ENV=1
bash recam_refine/run_step.sh shard-refine "$RECAM_ROOT" "$RECAM_WORK" \
  --shard-id 0 --worker-work-dir "$WORKER_A" \
  --devices 0,1,2,3,4,5,6,7 --gpu-batch-size 0
```

GPU B：

```bash
export RECAM_REFINE_REUSE_ENV=1
bash recam_refine/run_step.sh shard-refine "$RECAM_ROOT" "$RECAM_WORK" \
  --shard-id 1 --worker-work-dir "$WORKER_B" \
  --devices 0,1,2,3,4,5,6,7 --gpu-batch-size 0
```

这些命令不带 `--runtime-work-dir`：该参数属于原来的独立 runtime 模式。
如要复用原 GPU_RUNTIME 内的 Python，直接把 `RECAM_REFINE_PYTHON` 设为其 `runtime/env/bin/python`。
CPU 收尾的 shard-merge、apply、check、cleanup、repack 顺序不变，新 CPU 终端也要设置复用变量。

程序打印 `REUSING EXISTING ENVIRONMENT` 和完整路径。日志仍在各自工作目录，缓存和环境报告在
`runtime_cache/`；两个 GPU worker 不会共写缓存。GPU 依赖版本记录在协调目录
`existing_gpu_environment.json`，后续 worker 或续跑版本不一致时停止。
环境补装持有排他锁，后续运行持有共享读锁，锁覆盖实际子进程生命周期。
同一协调目录的 worker 不能在另一个 worker 运行期间补装。CPU 6a/6b 完成后再启动 GPU，避免在 GPU 节点等待下载。
运行期间不要从其他终端修改所用 Conda 环境；外部 pip/conda 安装器不受本脚本的锁控制。

## Hugging Face 连接错误

复用环境不等于修复 TLS 网络连接。程序保留当前 shell 的代理、CA 和动态库配置。
若已有原始 episode manifest，可给 transfer / align 加
`--episode-manifest /absolute/path/to/episode_manifest.jsonl`，或原始 chunk JSONL 目录，跳过相应映射下载。
必须使用这批 ReCam 的原始 episode 编号映射；不要用别的数据集重编号结果。
