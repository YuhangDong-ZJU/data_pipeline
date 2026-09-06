# ReCam 按功能逐步执行

以下每个处理命令只执行指定功能，成功或失败后都会停止。各步使用同一个 `RECAM_WORK`。
不要把本文所有代码块一次性粘贴执行；先执行当前步骤，查看结果，再选择下一步。

## 0. 对方已经 clone 过仓库：先更新代码

在对方机器原有的 `data_pipeline` 仓库目录中执行：

```bash
cd /absolute/path/data_pipeline
git status --short
git fetch origin
git switch codex/recam-dataset-refine
git pull --ff-only origin codex/recam-dataset-refine
```

首次切到该分支时，Git 会从 `origin/codex/recam-dataset-refine` 建立跟踪分支。
这组命令不包含 `reset --hard` 或删除文件。如果 Git 报本地修改冲突，先保留修改再继续，不要强制覆盖。

设置三个**真实的绝对路径**，不要直接使用下列占位路径：

```bash
export RECAM_ROOT=/absolute/path/recam_lerobot
export RECAM_WORK=/absolute/path/recam_refine_work
export DEPTH_OUTPUT=/absolute/path/droid_depth_output
```

目录要求：

| 变量 | 应包含的内容 |
| --- | --- |
| `RECAM_ROOT` | `simulation/`、`real_world/droid/` 等完整数据子集 |
| `RECAM_WORK` | 数据集外的工作目录，保存独立环境、断点、备份、报告和移出的日志；不能与其他两目录互相嵌套 |
| `DEPTH_OUTPUT` | droid-metric-depth 完成转换的输出根目录，含 `images/` 和 `annotations/foundation_stereo_depth/` |

处理时暂停会改写同一数据目录的下载、标注及训练任务。各模态跨文件更新期间不应作为完整训练集使用。
新开终端后重新设置同样的路径变量。不要更换或删除工作目录来重跑半成品。

## 1. 只迁移 metric depth

```bash
bash recam_refine/run_step.sh transfer "$RECAM_ROOT" "$RECAM_WORK" \
  --depth-output "$DEPTH_OUTPUT" --depth-chunks 2-13
```

执行内容：

- 校对所选 episode 的 UUID、相机 serial、FoundationStereo sidecar、帧数及完整 PNG。
- 所有选中流完成预检后，迁入 `real_world/droid/images/chunk-002` 到 `chunk-013` 的 `depth_01/02` 对应 episode/frame 路径。
- 已有冲突文件先备份；迁移完成核对逐文件 SHA-256；原 metadata 保持不变。
- 同盘优先移动，跨盘先复制并保留源 PNG，最终检查后的 `cleanup` 才清理源副本。

**此步不解 TAR、不裁尾、不处理 normal、不优化外参、不整理日志。** Wrist 不生成或迁移 depth/normal。
只安装固定版本 CPU 依赖，不需要提前安装 PyTorch，也不要求 normal 已准备好。

成功标志：`$RECAM_WORK/STEP1_DEPTH_TRANSFER_SUCCESS.json`，包括 episode 数、相机数、PNG 数。
日志：`$RECAM_WORK/step_transfer.log`；失败详情：`STEP1_FAILED.json`。

默认只下载相应 chunk 的原始映射。若已有自己的完整原始映射，在本步命令末尾加
`--episode-manifest /absolute/path/episode_manifest.jsonl`，第 3 步同样提供它。

## 2. 只解包 depth TAR

```bash
bash recam_refine/run_step.sh unpack "$RECAM_ROOT" "$RECAM_WORK"
```

恢复 simulation 和 real_world 子集的 depth PNG。DROID 已经在第 1 步迁入的新深度优先：
旧 TAR 仍检查完整性，但不会覆盖新 PNG，也不会恢复新流之外的旧尾帧。
未迁移的流正常解包，发现已有 PNG 与 TAR 内容冲突则明确停止。
忽略 DROID wrist depth TAR，最后整理时将其移出训练集。

**为保留恢复依据，本步暂时保留所有 TAR。** simulation TAR 最终也保留；
DROID 外部深度 TAR 在第 5b 步明确执行后删除，满足最终数据中不保留这些旧包的要求。

成功标志：`STEP2_UNPACK_SUCCESS.json`；逐包记录：`unpacked.json`；日志：`step_unpack.log`。

## 3. 只做 DROID 帧对齐、metadata 与来源修正

```bash
bash recam_refine/run_step.sh align "$RECAM_ROOT" "$RECAM_WORK" --workers 4
```

此时必须已经放齐两个外部相机的 NormalCrafter 输出：
`real_world/droid/videos/chunk-*/observation.images.normal_01/02/episode_*.mp4`。
原始 FoundationStereo sidecar 若还在其他目录，在命令末尾加：

```text
--depth-metadata /absolute/path/our_depth_output /absolute/path/another_depth_output
```

脚本自动读取 DROID 目录及第 1 步源输出目录中的 sidecar，不必重复列出它们。

以原始映射、当前 Parquet/meta、depth 数量、SVO 解码记录及全零深度尾帧确定共同前缀。
相对原始长度总共最多删除 2 个时刻；已经裁过的 episode 不重复裁掉额外帧，后补的 normal 也同步裁剪。
RGB（含 wrist）、外部 depth/normal、Parquet 所有字段、frame/index/timestamp、episode/global stats 和 provenance 一并对齐。
内部缺帧或无证据的非零重复画面不会被当作可随意删除的 padding。

修正外部 depth 来源为 FoundationStereo，normal 来源为 NormalCrafter，取消 wrist depth/normal 声明；
有 FS sidecar 时写入原生深度内参。**此步保留原始外参。** 原文件和移出的尾帧保存在工作目录。

成功标志：`STEP3_ALIGN_SUCCESS.json`，含最终帧数、此次删除时刻数。
详细计划：`plan.json`；数据内的逐 episode 记录：`meta/refinement.jsonl`；日志：`step_align.log`。

## 4a. 只统计与 PointWorld 的重合

```bash
bash recam_refine/run_step.sh overlap "$RECAM_ROOT" "$RECAM_WORK"
```

下载 PointWorld 相机 JSON 包（不下载其 TB 级其他数据），校对 UUID、两个物理相机 serial、
发布成功状态、合法矩阵和发布 loss 门槛。只输出报告，不改数据。

结果：`pointworld_overlap.json` 为各状态数量；`pointworld_overlap_episodes.jsonl` 为逐 episode 明细。
成功标志：`STEP4_OVERLAP_SUCCESS.json`；日志：`step_overlap.log`。
本地已有官方相机文件时加 `--pointworld-cameras /absolute/path/cameras`，后续步骤会记住这个目录。

## 4b. 只生成外参候选

```bash
bash recam_refine/run_step.sh refine "$RECAM_ROOT" "$RECAM_WORK" \
  --devices 0,1,2,3,4,5,6,7 --workers 4
```

这是首次需要 GPU 依赖的步骤；入口自动安装固定版本 PyTorch/CUDA 用户态库并检查可见 GPU，
不更改系统驱动、CUDA 或 Conda。目标为 Debian 12 / 8 × H100，不要求 nvcc 或 ZED SDK。

重合部分使用通过筛选的 PointWorld 发布外参；其余以 DROID 初值运行 PointWorld 机器人网格深度方法。
候选按拟合/选择验证指标筛选，默认失败且可观测的相机有有界重试；未通过的相机保留官方初值并记录原因。
这里的成功表示候选计算完整结束，不代表所有相机都优化成功或全数据质量已经验收。

**候选仅写入 `$RECAM_WORK/cameras/`，不改数据集外参。**
结果：`retained_official_calibrations.json`；成功标志：`STEP4_REFINE_SUCCESS.json`；日志：`step_refine.log`。

此时可以先用 [README 的外参审计、点云融合和 Viser 命令](README.md#可视化比较与-pointworld-质量指标)
对选定 episode 生成优化前后比较。审计从不可变 plan 读取原外参，从 `cameras/` 读取候选，
因此无需先写回。质量口径见 [CAMERA_QUALITY.md](CAMERA_QUALITY.md)。

## 4c. 明确写回候选外参

```bash
bash recam_refine/run_step.sh apply "$RECAM_ROOT" "$RECAM_WORK" --workers 4
```

确认候选内容与第 4b 步完成记录一致后，写入两个外部相机的外参，同步 metadata 和统计。
Wrist 外参、动作及状态仅保留此前对齐的时间前缀，不改其值。
此步不会重新优化，也不会再多裁尾帧。中断后继续从原始备份重建，避免在半成品上叠加变换。

成功标志：`STEP4_APPLY_SUCCESS.json`；日志：`step_apply.log`。

## 5a. 完整检查，检查完停止

```bash
bash recam_refine/run_step.sh check "$RECAM_ROOT" "$RECAM_WORK" --workers 4
```

检查所有子集的全部声明视频帧、PNG、Parquet/meta/stats 对齐、时间戳和保留字段摘要。
另对 DROID 的每个 episode 默认选择 8 个独立于本次拟合/候选选择的评估帧，
测量机器人深度残差、双视角 F1@5/20 mm 并比较前后质量；可加 `--audit-frames 24` 增加评估帧。
**完整解码检查是全帧检查，几何评估是逐 episode 的独立帧采样，两者不混称。**
没有 PointWorld 原作者的逐 episode 拟合帧清单，因此对直接复用的发布外参不声称证明了其训练/测试帧隔离。

结果：`checks.json`、`camera_audit/index.html`、`camera_audit/summary.json`。
全部通过才生成 `STEP5_CHECK_SUCCESS.json`，并记录训练文件状态摘要。
几何不可用、退步或不达门槛则返回非零状态，并保留失败明细；不会开启整理权限。
`camera_audit/COMPLETE.json` 仅表示测量完成，不能替代本步成功标志。

本步结束后 TAR、源副本和日志仍保留。可先审阅结果，再执行最后一步。

## 5b. 最后整理

```bash
bash recam_refine/run_step.sh cleanup "$RECAM_ROOT" "$RECAM_WORK"
```

只在第 5a 步成功、报告未变、训练文件状态未变时运行；暂停期间若改过训练数据，先重跑 `check`。

- 删除已经验证并解包的 DROID 外部 depth TAR；保留 simulation 和其他 real_world 的 TAR。
- 验证迁移 SHA-256 后清理跨盘保留的源 PNG，保留源输出目录中的 sidecar/log。
- 将旧 wrist depth/normal、旧 plural normals、日志、annotation/cache 等非训练内容移到工作目录。
- 再次确认声明的训练数据没有被整理操作改变。

最终成功标志：`SUCCESS.json`。原文件备份在 `original/`，辅助文件在 `auxiliary/`，
过时模态在 `unused_modalities/`，移出清单在 `moved_auxiliary.json`。这些备份不会自动删除。

## 查看进度、恢复与验证范围

随时查看各步的完成标志：

```bash
bash recam_refine/run_step.sh status "$RECAM_ROOT" "$RECAM_WORK"
```

完成的步骤再次执行会停止而不继续下一步；迁移在尚未开始裁尾时额外复核目标 PNG，
`check` 在最终整理前可以重新运行。失败详情保存在工作目录的 `*_FAILED.json` 和对应 `step_*.log`。
恢复使用原命令、原路径、原输入与原设置；输入内容改变时不会静默复用旧断点。
最终整理完如需再次检查，使用 README 中独立的只读 `check` / `audit-cameras` 命令。

旧自动入口 `run_refine.sh` 仍为兼容保留。**同一个 work_dir 不能混用自动与逐步流程。**
验证记录及真实样例质量边界见 [VALIDATION.md](VALIDATION.md)。
