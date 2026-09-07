# 将 CPU 阶段与 GPU 阶段分开执行

**机器已有环境时，先看 [复用已有环境](REUSE_ENVIRONMENTS.md)。** 设置 `RECAM_REFINE_REUSE_ENV=1` 后，
下列逐步处理命令使用已有环境；第 2 节的安装命令改为该文档中的环境检查，GPU 命令去掉 `--runtime-work-dir`。
本页保留原独立 runtime 方案供已经按该方案启动的作业使用。

GPU 机器只运行 `shard-refine`；其余阶段由 CPU 协调机完成。每次选择执行一步，成功后再继续。
单机方案的 `refine` 也需要 GPU，但与分片方案二选一。不要运行一键 `run.sh`，以免把整个流程留在 GPU 节点。

| 阶段 | 机器 | 主要工作 |
| --- | --- | --- |
| 更新代码、下载数据 | CPU | 网络和磁盘 |
| `transfer` | CPU | 校验、复制或迁移 metric depth |
| `unpack` | CPU | TAR 解包和 PNG 校验 |
| `align` | CPU | 裁尾帧、视频处理、Parquet/meta/stats 对齐 |
| `overlap` | CPU | 下载、校对 PointWorld 发布外参 |
| `shard-plan` | CPU | 固定分片、输入摘要、机器人资源准备 |
| 安装 CPU/GPU 环境 | CPU | 将锁定依赖安装到共享的独立目录 |
| `shard-refine` | GPU A/B | 批量外参优化，各运行一个分片 |
| `status` / `shard-status` | CPU | 查看状态 |
| `shard-merge` | CPU | 输入、候选和覆盖检查，合并 |
| `apply` | CPU | 写回外参及 metadata |
| `check` | CPU | 全媒体解码、数据一致性、逐 episode 独立帧几何评估 |
| `cleanup` | CPU | 检查门槛通过后整理非训练内容 |

`check` 使用 CPU PyTorch 计算几何指标，不需要 GPU、CUDA 或 NVIDIA 驱动。
它仍可能运行很久；默认 4 个 CPU/I/O worker，每个几何 worker 使用 2 个 Torch 线程。
不会为了降低 GPU 空闲时间而减少校验内容或降低质量门槛。

## 0. 共享路径和代码

CPU 协调机与 GPU A/B 都使用 Linux x86_64（支持 Debian 12），共享数据、协调目录、代码和 GPU 环境，
并挂载在相同绝对路径。共享 Python 环境需要允许执行文件。下面两个独立 worker 目录也放在共享存储，便于恢复。
所有节点设置下面的变量，替换为实际路径：

```bash
export REPO_DIR=/shared/data_pipeline
export RECAM_ROOT=/shared/recam_lerobot
export RECAM_WORK=/shared/recam_refine_work
export DEPTH_OUTPUT=/shared/droid_depth_output
export GPU_RUNTIME=/shared/recam_gpu_runtime_torch28
export WORKER_A=/shared/recam_workers/shard_0
export WORKER_B=/shared/recam_workers/shard_1
cd "$REPO_DIR"
```

保留已开始流程的原 `RECAM_WORK` 和 worker 目录；不要重建或清空记录。
这些工作目录和 `GPU_RUNTIME` 必须在数据集外，互不嵌套。共享存储要求见 [MULTI_MACHINE.md](MULTI_MACHINE.md)。
worker 目录也可以使用 GPU 本地磁盘；环境仍从共享 `GPU_RUNTIME` 读取，本地日志/断点需自行保留以供恢复。

仅在 CPU 节点更新共享仓库，不要多个节点同时操作 Git：

```bash
git status --short
git fetch origin
git switch codex/recam-dataset-refine
git pull --ff-only origin codex/recam-dataset-refine
git rev-parse HEAD
```

保留本地修改；分片开始后固定代码版本。先补齐需要下载的数据，再开始处理。
不要在处理期间重跑下载器覆盖已对齐的数据。

## 1. CPU：前处理

```bash
bash recam_refine/run_step.sh transfer "$RECAM_ROOT" "$RECAM_WORK" \
  --depth-output "$DEPTH_OUTPUT" --depth-chunks 2-13
```

```bash
bash recam_refine/run_step.sh unpack "$RECAM_ROOT" "$RECAM_WORK"
```

```bash
bash recam_refine/run_step.sh align "$RECAM_ROOT" "$RECAM_WORK" --workers 4
```

```bash
bash recam_refine/run_step.sh overlap "$RECAM_ROOT" "$RECAM_WORK"
```

```bash
bash recam_refine/run_step.sh shard-plan "$RECAM_ROOT" "$RECAM_WORK" \
  --num-shards 2 --refine-backend batched
```

`shard-plan` 不执行优化、不探测 GPU。不要因为运行在 CPU 机器上就加 `--devices cpu`：
该参数会选择 CPU reference 协议，和后续批量 GPU 优化不匹配。

## 2. CPU：提前安装全部环境

安装最终几何检查使用的 CPU 环境：

```bash
python3 recam_refine/bootstrap.py "$RECAM_WORK" --cpu-torch
```

只安装一套供 GPU A/B 共用的环境：

```bash
python3 recam_refine/bootstrap.py "$GPU_RUNTIME" --prepare-gpu
```

`--prepare-gpu` 安装锁定的 **PyTorch 2.8.0+cu129 / CUDA 12.9**，在 CPU 上检查导入和数值运算，
不要求 CPU 机器有 NVIDIA 驱动。无需修改系统 CUDA、Conda 或安装 nvcc。
它不能预先证明 GPU 节点的驱动可用；下一阶段会用实际 GPU 验证。
两台 GPU 共用 `$GPU_RUNTIME/runtime/env/bin/python`，CPU 检查环境使用独立的 **2.8.0+cpu**。
以上两条命令全部成功、`SHARD_PLAN_READY.json` 已生成后，再启动或申请 GPU 节点。

若之前准备过 2.5.1 GPU 环境，保留原目录，使用这里新命名的 `GPU_RUNTIME`；安装器拒绝原地替换不同版本的 GPU Torch。
本次环境适用于尚未开始外参优化的流程。已经用 2.5.1 开始优化的作业，应保留原代码/环境恢复，避免跨版本续接 Adam 断点或混合候选。

## 3. GPU：只运行优化

GPU A 重新设置第 0 步路径并进入仓库，然后执行：

```bash
bash recam_refine/run_step.sh shard-refine "$RECAM_ROOT" "$RECAM_WORK" \
  --shard-id 0 --worker-work-dir "$WORKER_A" \
  --runtime-work-dir "$GPU_RUNTIME" \
  --devices 0,1,2,3,4,5,6,7 --gpu-batch-size 0 --prepared-runtime
```

GPU B 同样设置路径，执行：

```bash
bash recam_refine/run_step.sh shard-refine "$RECAM_ROOT" "$RECAM_WORK" \
  --shard-id 1 --worker-work-dir "$WORKER_B" \
  --runtime-work-dir "$GPU_RUNTIME" \
  --devices 0,1,2,3,4,5,6,7 --gpu-batch-size 0 --prepared-runtime
```

`--runtime-work-dir` 选择共享环境，并强制只验证、不安装；`--prepared-runtime` 可保留以明确此意图。
入口核对已安装版本、依赖和实际 GPU 数值运算/CUDA Graph；不会下载或安装包。
环境缺失或版本不符时立即失败，回到 CPU 机器重新执行相应的准备命令。
GPU A/B 可以同时开始，不需要多机通信或分布式进程组。

共享环境的读锁保持到各 worker 结束，两台机器可以并发读取；安装器需要独占锁，运行期间重装会被拒绝。
`step_shard-refine.log`、`calibration_state/` 和 `runtime_cache/` 仍分别写入各自的 `WORKER_A/B`。
已有 worker 内的旧 `runtime/` 不会被覆盖或删除；新命令选择指定的共享环境，原日志、断点及候选仍沿用。
旧的独立环境入口仍然兼容，省略 `--runtime-work-dir` 即保持原行为。

CPU 协调机查看进度：

```bash
bash recam_refine/run_step.sh shard-status "$RECAM_ROOT" "$RECAM_WORK"
```

各 worker 命令成功退出、对应分片状态为 `complete` 后，该 GPU 节点即可释放。
CPU 合并阶段继续核对完整覆盖和结果摘要。共享 worker 目录须保留，以便有需要时恢复。
输入读取、断点写入和最后不足一批仍可能有短暂 GPU 空闲；不承诺利用率始终 100%。

## 4. CPU：合并、写回、检查、整理

确认所有分片完成后，每次只执行一条：

```bash
bash recam_refine/run_step.sh shard-merge "$RECAM_ROOT" "$RECAM_WORK"
```

```bash
bash recam_refine/run_step.sh apply "$RECAM_ROOT" "$RECAM_WORK" --workers 4
```

```bash
bash recam_refine/run_step.sh check "$RECAM_ROOT" "$RECAM_WORK" \
  --workers 4 --audit-frames 24 --prepared-runtime
```

`check` 全帧解码所有声明媒体，这里将几何评估设为每 episode 24 个独立采样帧。
通过后生成 `STEP5_CHECK_SUCCESS.json`；质量需复核或任何检查失败时停止，并保留报告。
报告在 `$RECAM_WORK/checks.json` 和 `$RECAM_WORK/camera_audit/index.html`。
对 PointWorld 发布外参，无法证明评估帧未被原作者用于拟合，详见 [CAMERA_QUALITY.md](CAMERA_QUALITY.md)。

检查通过且数据未再更改后：

```bash
bash recam_refine/run_step.sh cleanup "$RECAM_ROOT" "$RECAM_WORK"
```

```bash
bash recam_refine/run_step.sh status "$RECAM_ROOT" "$RECAM_WORK"
ls -l "$RECAM_WORK/STEP5_CHECK_SUCCESS.json" "$RECAM_WORK/SUCCESS.json"
```

阶段、完成标志和恢复规则与 [STEP_BY_STEP.md](STEP_BY_STEP.md) 相同。
可视化使用现有候选时也可在 CPU 上执行；不要加会重新拟合的 `--fit`。

## 5. CPU：可选，重新打包最终 DROID depth

`cleanup` 成功后，按原发布布局重新生成两个外部相机的 depth TAR：

```bash
bash recam_refine/run_step.sh repack "$RECAM_ROOT" "$RECAM_WORK" \
  --episodes-per-shard 250 --workers 4 --prepared-runtime
```

脚本自动检查前置完成记录及数据状态，逐文件验证 TAR 后才发布，原 PNG 保留用于训练。
看到 `REPACK COMPLETE` 表示打包和核验成功。空间、恢复和文件位置见 [REPACK_DEPTH.md](REPACK_DEPTH.md)。
