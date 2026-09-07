# 共享数据目录：分片、分机器执行外参优化

完成 [逐步流程](STEP_BY_STEP.md) 的 `transfer → unpack → align → overlap` 后，
用本文的 `shard-plan → shard-refine → shard-merge` 替代单机 `refine`。
每条命令完成后停止；最后仍由主机逐步执行 `apply → check → cleanup`。

## 目录和版本

两台机器共享 **同一份数据目录和同一个协调工作目录**。无需复制数据、切分 Parquet 或重新编号 episode。
PointWorld 重合部分直接保存发布外参，其余 episode 按 UUID 稳定打散并均分；每个 episode 只属于一个分片。
分片内部沿用每卡常驻进程、动态批量、预读和 CUDA Graph，优化目标、迭代预算、质量门槛不变。

在两台机器各自已有的仓库中，先更新到相同版本：

```bash
cd /absolute/path/data_pipeline
git status --short
git fetch origin
git switch codex/recam-dataset-refine
git pull --ff-only origin codex/recam-dataset-refine
git rev-parse HEAD
```

两边最后的提交号应相同。保留本地修改，遇到 Git 冲突不要强制覆盖。开始分片后不再更新处理代码。
在两台机器的终端分别设置实际路径：

```bash
export RECAM_ROOT=/shared/recam_lerobot
export RECAM_WORK=/shared/recam_refine_work
```

上述只是路径示例。`RECAM_WORK` 必须是前面步骤使用的原工作目录，保存了计划、备份和处理记录；
不能改成一个空目录、复制出第二份协调目录或移动已有记录。它必须位于数据集外，且与数据集互不嵌套。
两台机器将共享目录挂载到相同绝对路径，确保协调目录中的独立 Python 环境也可直接访问。
前面步骤及最后的合并、写回、检查、整理在原主机、原路径执行。

每个分片再使用独立的 `--worker-work-dir`，例如本机可写的 `/scratch/recam_worker_0`。
它保存运行环境、日志、Adam 断点和候选缓存；推荐本地 SSD，不能放在数据集或 `RECAM_WORK` 内，三者互不嵌套。
入口自动安装锁定环境；无需额外安装 CUDA Toolkit、模型权重或多机通信库。

## 1. 主机生成固定分片计划

```bash
bash recam_refine/run_step.sh shard-plan "$RECAM_ROOT" "$RECAM_WORK" --num-shards 2
```

只读取数据，下载并校验共享的机器人资源，保存 PointWorld 发布候选、分片清单及输入摘要。
摘要覆盖每个待优化 episode 的完整 Parquet、实际采样的两个外部相机 PNG，以及相关 metadata。
首次规划会产生共享存储读取开销，日志显示摘要进度。

成功标志：`$RECAM_WORK/SHARD_PLAN_READY.json`。
清单：`$RECAM_WORK/shards/manifests/shard-00000.json`、`shard-00001.json`。
计划：`$RECAM_WORK/shards/plan.json`，记录各分片的原始 episode 编号和总覆盖范围。

分片数、输入、算法代码、后端和迭代预算在这一步固定。默认预算仍是 2,000 次，失败且可观测的相机续到总计 6,000 次。
如果单机 `refine` 已经开始，程序拒绝混用；保留原工作目录并恢复原单机命令。

## 2. 两台机器分别运行各自分片

机器 A，分片 0（替换本机可写的 worker 路径）：

```bash
bash recam_refine/run_step.sh shard-refine "$RECAM_ROOT" "$RECAM_WORK" \
  --shard-id 0 --worker-work-dir /scratch/recam_worker_0 \
  --devices 0,1,2,3,4,5,6,7 --gpu-batch-size 0
```

机器 B，分片 1：

```bash
bash recam_refine/run_step.sh shard-refine "$RECAM_ROOT" "$RECAM_WORK" \
  --shard-id 1 --worker-work-dir /scratch/recam_worker_1 \
  --devices 0,1,2,3,4,5,6,7 --gpu-batch-size 0
```

两条命令可以同时执行。GPU 编号是各自机器上的编号，两台机器可以使用不同数量的 GPU。
`--gpu-batch-size 0` 根据各卡空闲显存自动选批量；也可以明确设置批量。worker 自动继承计划的迭代预算，
无需再传 `--iterations`；显式传入不同预算会报错。

**worker 只读数据集，只生成外参候选。** 同一分片的重复 worker 被锁拒绝；不同分片可并行。
worker 完成时生成各自的 `$RECAM_WORK/shards/results/shard-0000N/COMPLETE.json`，
不会生成整个 refine 的完成标志，也不会自动写回外参或整理文件。

更多分片也支持：规划时设 `--num-shards N`，每次运行一个 `--shard-id 0..N-1`。
两台机器可各自顺序处理多个分片，每个分片使用自己的 worker 目录。
全部计划分片（包括空分片）都需运行完成后合并；任务运行中不修改分片数。

## 3. 查看进度和恢复

任一机器执行：

```bash
bash recam_refine/run_step.sh shard-status "$RECAM_ROOT" "$RECAM_WORK"
```

输出每个分片的状态、episode 数、最近进度及是否已合并。`computing` 是最近一次记录，
不能证明远端进程仍存活；异常断电后用原命令恢复即可，进程退出后系统释放文件锁。

每台机器查看自己的日志，例如机器 A：

```bash
tail -f /scratch/recam_worker_0/step_shard-refine.log
```

- worker 目录的 `calibration_performance.jsonl`、`calibration_timing.json`：逐批和本次运行耗时。
- worker 目录的 `calibration_state/`：每 1,000 次保存的相机级 Adam 断点。
- 共享 `shards/results/shard-0000N/progress.json`、`FAILED.json`：最近进度或错误。
- 共享 `shards/plan_timing.json`、最终合并记录：规划摘要和合并阶段的独立耗时。

中断后重跑同一条命令，保留原 worker 目录；允许调整 GPU 列表和批量。已完成候选不重复优化，
未完成相机从匹配输入的断点恢复。若换机器且无法访问原本地 worker 目录，可给同一分片指定新的独立目录：
已发布到共享结果目录的候选可以复用，尚未发布的本地候选和 Adam 状态不会自动跨机器转移。
完成的分片再次执行会验证候选后返回。

## 4. 主机验证并合并全部分片

确认所有分片完成后，回到原主机执行：

```bash
bash recam_refine/run_step.sh shard-merge "$RECAM_ROOT" "$RECAM_WORK"
```

合并检查计划和代码版本、输入字节、episode/UUID、候选来源和接受门槛、分片完成摘要及全量覆盖；
拒绝缺片、重复、额外候选、结果被改写或数据在运行期间改变。不会把一个分片误当作全量结果。
全部验证后才发布 `$RECAM_WORK/cameras/` 和 `STEP4_REFINE_SUCCESS.json`，然后停止。
合并中断可重跑同一命令，不需要手工复制 JSON。

之后与单机步骤完全相同，仍然每次只选择执行一步：

```bash
bash recam_refine/run_step.sh apply "$RECAM_ROOT" "$RECAM_WORK" --workers 4
```

```bash
bash recam_refine/run_step.sh check "$RECAM_ROOT" "$RECAM_WORK" --workers 4
```

```bash
bash recam_refine/run_step.sh cleanup "$RECAM_ROOT" "$RECAM_WORK"
```

合并只证明候选计算和覆盖完整，最终几何质量仍由 `check` 的独立帧评估及全媒体检查决定。
检查失败时保留数据和报告，不允许整理。合并后也可先按 [可视化说明](README.md#可视化比较与-pointworld-质量指标)
查看优化前后点云，再决定何时 `apply`。

## 共享存储与验证范围

共享存储须支持跨机器 POSIX 文件锁及原子重命名。所有处理命令使用同一共享协调目录：
worker 持共享锁，写操作持独占锁，避免同时处理中的跨文件更新。
程序会拒绝已知禁用远端 flock 的 NFS 挂载选项（`nolock`、`local_lock=all/flock`），
以及不支持本流程共享锁保证的 SSHFS/rclone/s3fs 挂载。
NFS 的目录独占锁存在平台限制，因此协调锁使用可写的普通文件；参见 [Linux flock 文档](https://man7.org/linux/man-pages/man2/flock.2.html)。
这些锁保护使用本工具的进程，下载、标注或其他直接改写数据的程序仍需暂停。

验证覆盖临时完整数据的 Bash 命令、缺片/冲突/修改输入拒绝、恢复、合并及显式写回边界；
另用真实样例的独立副本，运行两个独立 GPU 进程完成 4 个 episode / 8 个相机，
候选与单机相同输入运行的矩阵差异为 0，源文件摘要前后一致。
这是同一服务器、两张 RTX 4090、共享本地文件系统的验证，**尚非两台物理机器、真实 NFS 或 H100 的吞吐量实测**。
详细验证见 [VALIDATION.md](VALIDATION.md)，计时边界见 [PERFORMANCE.md](PERFORMANCE.md)。
