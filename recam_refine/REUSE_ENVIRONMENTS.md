# 复用机器上已有的环境

已有环境可以直接用于处理，不必运行 `bootstrap.py`。复用模式只查找、导入和测试已有依赖，
不执行 conda create、pip install、升级或降级。它支持 Python 3.10 / 3.11；原锁定 runtime 的用法继续保留。

每次新开 CPU 或 GPU 终端，设置原来的数据、工作目录变量，再加：

```bash
export RECAM_REFINE_REUSE_ENV=1
```

随后 `run_step.sh` 的每一步都使用复用模式。也可以不设置这个变量，在每条命令末尾加 `--reuse-env`。
`--prepared-runtime` 可以保留，但不再触发旧版锁定环境的安装或版本核对。

## 自动查找规则

Conda 位置按 `DATA_PIPELINE_CONDA_BIN`、`DROID_NORMALS_CONDA_BIN`、`DROID_DEPTH_CONDA_BIN`、
`MINIFORGE_HOME/bin/conda`、PATH、家目录下的 miniforge3/miniconda3 查找。
已有的 `TRIM_ENV_NAME`、`DROID_NORMALS_ENV_NAME`、`RECAM_DOWNLOAD_ENV_NAME` 覆盖设置继续生效。

CPU 优先测试 `recam_data_pipeline`，然后 `droid_normals`、`recam_download`；GPU 优先测试 `droid_normals`。
之后尝试当前工作目录已准备的 `runtime/env/bin/python`、激活的 Conda 环境及 PATH 中的 Python。
每个候选都会实际验证；名称相同不等于依赖已经满足。

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

进入已更新的代码仓库后，执行下面的只读环境检查。它们只向 RECAM_WORK 写报告、缓存，
不修改数据集或现有环境。CPU 节点无需 GPU 或 NVIDIA 驱动。

```bash
# 迁移、解包、对齐等基础能力
python3 -m recam_refine.environment "$RECAM_WORK" --profile base

# 最终 CPU 校验能力；允许复用 2.8.0+cu129，在 CPU 上运算
python3 -m recam_refine.environment "$RECAM_WORK" --profile cpu

# 提前检查将供 GPU 使用的环境；此时只执行 CPU 运算，不探测 GPU
python3 -m recam_refine.environment "$RECAM_WORK" --profile prepare-gpu
```

这里代替原流程的两条 `bootstrap.py --cpu-torch` / `--prepare-gpu` 安装命令。
报告中的 `python` 是实际选中的解释器。GPU 节点需能访问该已有环境；如各节点 PATH/Conda 配置不同，
在两台 GPU 节点用 `RECAM_REFINE_PYTHON` 指向 CPU 检查通过的同一路径。

基础检查包括依赖导入、16 位 PNG 读写、Parquet 读写以及 H.264 编解码。
几何步骤还检查机器人网格采样。CPU 校验要求 PyTorch 2.8.0+cpu 或 2.8.0+cu129；
GPU 预检查和优化保持本次约定的 2.8.0+cu129，GPU 优化启动时还验证实际 CUDA 运算和批量优化器。
不会因为换环境降低原来的外参质量门槛。

没有候选满足要求时，输出每个环境缺少/不可用的依赖并停止。已有的三个 Conda 环境并不保证
包含 refine 新增的全部依赖；复用模式不会为补齐它们而自动改动旧环境。已准备好的 refine runtime 也能复用。

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
运行期间不要从其他终端修改所用 Conda 环境；外部 pip/conda 安装器不受本脚本的锁控制。

## Hugging Face 连接错误

复用环境不等于修复 TLS 网络连接。程序保留当前 shell 的代理、CA 和动态库配置。
若已有原始 episode manifest，可给 transfer / align 加
`--episode-manifest /absolute/path/to/episode_manifest.jsonl`，或原始 chunk JSONL 目录，跳过相应映射下载。
必须使用这批 ReCam 的原始 episode 编号映射；不要用别的数据集重编号结果。
