# 按机器执行（Linux Bash）

本页将飞书 CPU 0a–6b、GPU A/B、CPU 7–12 合并为四个入口。路径已按飞书配置，无需逐段 export，也无需手动填写扫描报告或共享 Python 路径。

| 机器 | 入口 | 自动顺序 |
|---|---|---|
| CPU，先执行 | `run_prepare.sh` | 自动拉取最新代码并重新进入新版 → 复用并补齐一套共享环境 → 自动定位已有报告或扫描 chunk 2–13 → 排除已确认的源 episode 6795 → 迁移 → 解压 → 对齐 → PointWorld 重合统计 → 两个固定分片；已完成的数据步骤跳过 |
| GPU 机器 1 | `run_gpu1.sh` | 只检查已准备环境，运行分片 0，使用本机可见 GPU 0–7 |
| GPU 机器 2 | `run_gpu2.sh` | 只检查已准备环境，运行分片 1，使用本机可见 GPU 0–7 |
| CPU，两台 GPU 成功后 | `run_cpu_finish.sh` | 合并 → 写回外参 → 全量校验与每 episode 最多 24 帧几何审计 → 清理 → 每 TAR 最多 250 个 episode 的 DROID 外部相机 depth 打包 |

GPU1/GPU2 指两台机器，不是单张显卡。两台 GPU 可同时运行。入口不会自动开机、SSH 或跨机器启动任务。

默认并发：迁移 8 路（`TRANSFER_WORKERS`），TAR 解压 8 路（`UNPACK_WORKERS`），JSON 首次读取 8 路，对齐 8 个进程，外参写回及最终校验 4 个进程，TAR 打包 8 路。迁移按相机序列隔离任务，解压按相机目录隔离；同一目录下的归档串行处理。I/O 任务队列最多保留并发数的两倍，不一次提交所有文件。状态只由主线程写入，迁移日志显示相机序列完成数/总数和已处理 PNG 数。

四个入口会自动使用这些默认值。需要调整迁移或解压并发时，在运行前设置 `export TRANSFER_WORKERS=16` 或 `export UNPACK_WORKERS=4`；并发数只改变调度，不清空已有记录，也不改变完成步骤的判断。共享存储吞吐不一定随并发数线性增长。

迁移不再有独立预检阶段：每个相机序列核对文件名、数量和目标路径后立即迁移，不先扫描全部 episode，也不抽查 PNG 或逐文件读取大小。目录移动后只核对文件名清单，内容解码统一在最终校验执行。迁移前不再读取全部深度 JSON；相机身份、时间戳与 padding 检查统一在 align 中完成，缺失或异常的迁移 sidecar 会阻止对齐成功。同文件系统直接移动目录，不做迁移前全量 PNG 哈希扫描；跨文件系统直接复制，检查写入字节数和源文件属性是否变化；删除源副本前再逐块比较。完成迁移后不重复全量读取 PNG。身份清单优先复用本地缓存。最终全量文件有效性和模态一致性校验仍在 CPU 收尾执行。

align 将已校验 JSON 的紧凑结果存入 `$RECAM_WORK/input_cache/validated_depth_json_v1.sqlite3`。重跑只核对文件属性，复用未变化的结果；新增或变化的 JSON 使用 8 路 I/O 读取。缓存每 64 个未命中条目提交一次，正常异常退出也保存已校验条目；强制终止可能重读最后尚未提交的小批次。旧版只有内存中的 JSON 结果无法恢复，但新版 transfer 不再等待这批读取。align 首次需要建立缓存，之后复用；不要删除工作目录或手工编辑缓存。

新版兼容原工作目录中的旧迁移记录。已完成步骤继续跳过，未完成迁移按原记录恢复；不要删除记录或重新排除 episode。更新前先停止正在运行的脚本并等待退出，再重新运行 `bash run_prepare.sh`。不要在两台 GPU 或其他处理任务运行时更新共享代码。

旧 TAR 中已被迁入 metric depth 替代的成员直接跳过，不生成临时 PNG；其他成员照常解压。仿真 TAR 保留。

首次解压 TAR 直接写出成员，不计算归档或成员内容哈希，也不解码 PNG。已完成归档保存输入和输出的文件属性，重跑属性未变化时跳过解压。裁剪视频只执行裁剪写出，统一由最终校验检查帧数、PTS 和解码。新 TAR 直接打包，不估算占用，也不查询剩余磁盘空间，不计算内容哈希，也不在写完后重新读取整包；检查写入长度并保留原 PNG。已完成且文件属性未变化的 TAR 自动跳过；只有缺少完成记录的已有 TAR，才逐成员比较以恢复断点。已有工作目录、完成标记和备份继续沿用。

收尾校验中，每张 PNG 从存储读取一次，在内存直接解码并检查格式和数值，不额外运行 PNG verify/CRC 扫描。清理前保留训练文件变更摘要检查，但不再重新全量计算迁移 PNG 的哈希；清理结束不重复遍历训练文件。跨文件系统源副本删除前仍逐块比较源和最终文件，按相机序列使用 8 路并发。视频逐帧解码、模态对齐、metadata/stats 和外参审计保留，不将抽样报告标记为全量校验通过。

## 1. 仅当旧仓库还没有新版入口时：首次获取入口

```bash
cd /mnt/bn/yuyingchen/moranli/Code/Research/ModelArch/data_pipeline
git status --short
git fetch origin &&
git switch codex/recam-dataset-refine &&
git pull --ff-only origin codex/recam-dataset-refine
```

以上成功后再执行下面命令。这段只用于取得支持自动更新的新版入口；之后无需手动执行 Git 命令。保留本地修改；不要 reset/clean。

## 2. CPU：一次完成所有准备

```bash
cd /mnt/bn/yuyingchen/moranli/Code/Research/ModelArch/data_pipeline
bash run_prepare.sh
```

每次启动先取得工作流独占锁，自动 `git fetch`、切换到 `codex/recam-dataset-refine`、`git pull --ff-only`，然后重新进入刚拉下来的入口，保留锁和日志。网络失败、非快进或存在未提交的已跟踪文件修改时停止，不继续用旧代码处理数据，不会强制覆盖本地内容。GPU 正在运行时无法取得独占锁，因此不会更新共享代码。

更新后保留全部成功记录与断点；原路径和分片计划必须一致。若处理代码摘要发生变化，在 CPU 上复核共享环境要求、仅补齐缺项，并将版本变更记录到 `$RECAM_WORK/launchers/code_updates.jsonl`，再逐项跳过已完成的数据步骤。代码无变化且准备已完成时，直接提示跳过。GPU 和收尾入口不自行拉代码；需要更新时，先停止所有任务，再运行 `run_prepare.sh`。

优先选择已有 PyTorch 2.8.0+cu129 环境，补装缺项，并将 CPU 校验依赖补齐到同一解释器；保留已有包版本，遇到不兼容版本停止。CPU 无需 NVIDIA 驱动。准备结果存入 `$RECAM_WORK/launchers/PREPARE_READY.json`。所有机器须能访问其中记录的相同绝对路径。

扫描只检查 sidecar，不能证明每张 PNG 的时序。只有最新报告明确对应指定路径、chunk 2–13、唯一异常 6795 才继续排除；不会因最新报告不符合而回退旧报告。排除脚本仍重新核对原 sidecar，并保留所有相关备份。

## 3. 两台 GPU 分别执行

GPU 机器 1：

```bash
cd /mnt/bn/yuyingchen/moranli/Code/Research/ModelArch/data_pipeline
bash run_gpu1.sh
```

GPU 机器 2：

```bash
cd /mnt/bn/yuyingchen/moranli/Code/Research/ModelArch/data_pipeline
bash run_gpu2.sh
```

自动使用准备阶段记录的 Python，不重新选择、安装或更新环境。实际 CUDA 检查仍在 GPU 主机执行。环境缺失时停止，先修复共享挂载或在任务停止后处理依赖。GPU 阶段仍会进行必要的输入摘要检查和 CPU 数据加载。

## 4. CPU：一次完成收尾

```bash
cd /mnt/bn/yuyingchen/moranli/Code/Research/ModelArch/data_pipeline
bash run_cpu_finish.sh
```

分片不完整时合并失败并停止，不会写回。校验失败时不清理、不打包。重跑入口会跳过已成功的整步校验，清理本身仍核对校验报告和训练文件摘要，防止使用过期结果；提示摘要变化时应停止并定位，不要手动伪造或删除成功记录。最后 DROID depth 保留 PNG 和已验证的 TAR，simulation 保留 PNG 和原 TAR。

## 查看、恢复与配置

只看命令，不安装环境、不改数据：

```bash
bash run_prepare.sh --dry-run
bash run_gpu1.sh --dry-run
bash run_gpu2.sh --dry-run
bash run_cpu_finish.sh --dry-run
```

报错后修复原因，再运行同一入口。已有旧版逐步执行记录也会被识别；若数据已经推进到后续阶段，不重新执行排除。中途不要切换路径、改变固定参数或重新分片。首次准备完成后冻结路径、配置及分片计划；更新代码统一通过所有任务停止后的 `run_prepare.sh` 完成，GPU 入口仍拒绝未经准备入口复核的代码变更。

可以从 `run_prepare.sh` 开始依次重跑四个入口：已成功的步骤打印 `SKIPPED ... 已完成，因此跳过`，未完成的步骤恢复执行。准备全部完成时逐项提示跳过；即使 CPU 收尾已完成，GPU 入口也会在核对原分片记录与候选结果哈希后跳过，不要求重新初始化 GPU。记录或结果不一致时明确报错，不把“存在文件”误当作可跳过的成功结果。

统一默认配置在 `recam_refine/launchers/config.sh`，可在首次准备前通过同名环境变量覆盖；后续机器必须使用同样配置。无需修改 `REPO_DIR`，入口自动定位自己的仓库目录。

总日志：`$RECAM_WORK/launchers/run_prepare.log`、`run_gpu1.log`、`run_gpu2.log`、`run_finish.log`。原各步骤日志继续保留；GPU 详细日志位于各自 worker 目录。

```bash
tail -f /mnt/bn/pistis/moranli/Data/recam_lerobot/recam_refine_work/launchers/run_prepare.log
```

总日志和原步骤日志均位于训练数据集之外。新入口之间有工作流锁：CPU 准备/收尾与 GPU 运行不能同时进行；两台 GPU 可并行，各分片另有重复启动保护。不要混用旧单步命令与正在运行的新入口来绕过工作流锁。
