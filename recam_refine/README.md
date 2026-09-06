# ReCam 全量恢复与 DROID refine

入口：`recam_refine/run_refine.sh`。目标环境为 **Debian 12、Linux x86_64、8 × H100**。
无需 sudo、Conda、Docker、nvcc、ZED SDK、FoundationStereo/NormalCrafter 模型权重。
脚本直接使用已经生成的深度 PNG、normal MP4 和 Parquet 中的机器人状态。

## 对方服务器的执行命令

先结束对这个数据目录的下载、深度/normal 标注和训练任务。处理期间数据集必须离线；
多文件更新发生中断时，用**同一条命令和同一个 work_dir**恢复。

在仓库根目录执行下面一条命令，替换三个绝对路径：

```bash
bash recam_refine/run_refine.sh \
  /absolute/path/recam_lerobot \
  /absolute/path/recam_refine_work \
  --depth-output /absolute/path/droid_depth_output
```

三个目录的含义：

| 参数 | 内容 |
| --- | --- |
| `recam_lerobot` | 包含 `simulation/*` 和 `real_world/droid` 等 LeRobot 子集 |
| `recam_refine_work` | **必须位于 recam_lerobot 外**；保存独立运行环境、校验报告、日志、原文件备份和移出的辅助文件 |
| `--depth-output` | 对方运行 droid-metric-depth 得到的输出根目录，下面有 `images/` 和 `annotations/foundation_stereo_depth/` |

默认转移 **chunk-002 至 chunk-013（含两端）**，使用 GPU `0,1,2,3,4,5,6,7`，CPU/I/O 并发数为 4。
normal 应已在 `real_world/droid/videos/chunk-*/observation.images.normal_01|02/` 下。
不会处理 wrist 的 depth/normal；wrist RGB 和逐帧外参保留。

若你们处理的其他 chunk 还有 FoundationStereo 原始 sidecar，请一起提供：

```bash
bash recam_refine/run_refine.sh \
  /absolute/path/recam_lerobot \
  /absolute/path/recam_refine_work \
  --depth-output /absolute/path/droid_depth_output \
  --depth-metadata /absolute/path/our_depth_output /absolute/path/another_depth_output
```

这是数据路径配置，不需要修改 Python 或 Bash 代码。源 PNG 已经转移完毕且无需步骤 1 时可省略 `--depth-output`。
若 normal 缺失，脚本在解包/转移之前给出具体路径并退出，不会冒充已经完成 normal 标注。

## 环境与输入

- 安装器用系统 `python3` 的标准库下载并校验固定版本 uv；在 work_dir 中安装 CPython **3.11.11**。
- CPU 和 GPU 依赖分别固定在 `requirements.lock` / `requirements-gpu.lock`，安装时校验 SHA-256，只接受 wheel。
- PyTorch **2.5.1 + CUDA 12.4** 自带 CUDA 用户态运行库，支持 H100 的 `sm_90`；系统只需可用的 NVIDIA 驱动。
- 安装后检查 PyAV/libx264、Parquet，并在每张可见 GPU 上执行实际优化所需的 `grid_sample` 前向/反向。
- 不改系统 Python、CUDA、驱动或已有 Conda 环境。建议 work_dir 为运行环境预留约 10 GB，另加处理备份空间。
- 首次运行需要访问 GitHub、PyPI、PyTorch wheel 源和 Hugging Face。网络不可用时明确退出；重跑会复用下载缓存。

脚本自动下载两个小型输入，**不会下载 PointWorld 的 TB 级 flow 数据或重新下载 SVO**：

1. `Sponbebob4258/droid-24k-external-svo` 的 chunk manifest，用于 ReCam episode index → DROID UUID / camera serial 映射；
2. `nvidia/PointWorld-DROID` 的相机 JSON 压缩包，约 20 MB 下载量。

Hub revision 在第一次运行时解析并固定在 `inputs/hub_revisions.json`，恢复时继续使用相同 revision。
PointWorld 的 URDF/mesh 固定到 commit `3872ec6ee73146aa671192ef79b5dfbedc0246e3`。
已有本地输入时可以覆盖下载：

```bash
# 追加到上面的 run 命令
--episode-manifest /absolute/path/episode_manifest.jsonl \
--pointworld-cameras /absolute/path/pointworld/droid/cameras
```

`episode_manifest` 必须是这批 ReCam 的原始映射，不能使用重新编号的其他 DROID 数据集的映射。
脚本核对 UUID、相机 serial、长度、可用时的任务文本，并拒绝冲突的 sidecar。

## 处理顺序与结果

1. **恢复 TAR**：识别原 `pack_depth.py` 的 subset-root 相对路径，逐 PNG 校验后原子安装；拒绝路径穿越、链接、重复成员、非深度文件、损坏 PNG 和已有内容冲突。simulation 的 TAR 保留；DROID TAR 到全量校验完成后才删除，防止旧包复活已裁掉的尾帧。其他 real_world 子集的 TAR 保留。
2. **接收 chunk 2–13**：只接收 external_1/2，核对 FoundationStereo sidecar。相同文件系统优先原子重命名，避免再复制数 TB；已有目标冲突先备份。跨文件系统复制并核对 SHA-256，全部校验后才删除源 PNG。源日志/sidecar 保留在源输出目录。
3. **确定共同有效前缀**：以当前 Parquet/meta 长度、所有 depth 长度、SVO `decoded_frame_count` 和全零深度尾帧共同决定终点，每个 episode 相对原始 manifest 总共最多删除 2 帧。已经裁过的 episode 不会重复裁剪；后补 normal 仍会独立对齐。缺少历史 sidecar 时，只把末尾最多 2 张全零深度认定为“无效深度尾帧”，不会将其伪称为已证明的 SVO padding。重复的非零静止画面不会被删除；内部缺帧、内部全零深度、超过 2 帧的短缺会报错。
4. **外参**：按 source UUID 和物理 camera serial 匹配。PointWorld 发布的 robot-base → OpenCV camera 矩阵先求逆，再写入 ReCam 的 camera → robot-base 字段；发布状态必须成功、矩阵合法、final loss < 0.10。其余相机使用 DROID 初值，按 PointWorld 的机器人网格投影与 FoundationStereo 深度 L1 对齐方法优化。使用 Panda + Robotiq 2F85 网格、关节正运动学以及 `gripper_position * 0.725`，保留逐相机内参。与原方法的变化是跳过 VGGT 初始化，增加独立验证帧、最佳迭代选择和位姿变化范围检查。**不可观测或验证误差变差时保留官方初值**，记录在 `retained_official_calibrations.json`，不会把失败优化写入数据集。真正的运行异常会停止流程并保留断点。
5. **对齐写入**：所有声明的 RGB/normal 视频、depth PNG、Parquet 的全部字段保留同一时间前缀；全局 `index` 连续重建。RGB/normal 按解码后的显示顺序裁剪，用无损 H.264 保留源解码像素，并逐帧比较前缀。**裁剪后的视频可能变大**，需要为目标文件和备份留空间。不会以 packet 数量冒充帧数，也不会把裁剪后的最后一步标成虚构的任务成功。
6. **元数据**：重建 `info.json`、`episodes.jsonl`、全部数值字段的 episode/global stats；修正 `cameras.json` 的 FoundationStereo/NormalCrafter 来源。只声明两个外部 depth/normal。NormalCrafter 保持模型原生 view-space 约定，不沿用旧 Open3D 法向的坐标声明。新增 `meta/refinement.jsonl` 保存 UUID、serial、最终长度、裁剪依据、外参来源、质量指标和动作/状态/wrist 原值校验摘要。原 episode ID、任务 catalogue、split 和训练所需坐标元数据保留。
7. **全量校验与整理**：解码所有声明的视频帧、读取所有 PNG，核对帧数、PTS、FPS、分辨率、位深、深度范围、索引、时间戳、外参矩阵、任务引用和统计值；再次验证动作/状态/wrist 仅被切前缀、没有被改值。**所有子集通过后**才删除 DROID TAR、清理跨盘复制的源 PNG，并把旧 wrist depth/normal、旧 plural `normals_*` 和 log/annotation/cache 文件移到 work_dir。

simulation 和其他 real_world 子集的原始媒体与标注方法保持原样，恢复 depth 后参与全量校验。
检查是完整解码，不是抽查；完整 ReCam 的 I/O 和外参优化耗时可能很长，不能由少量样例推断全量运行时间。

## 完成、恢复与单独检查

完成标志是 `<work_dir>/SUCCESS.json`。其中 `full_decode` 表示结构/文件检查完整通过，
`all_external_calibrations_accepted` 表示是否所有外参候选都通过几何验收。
若后者为 false，请查看 `retained_official_calibrations.json`：这些相机仍使用原有官方标定，不能称为全部优化成功。
结构检查通过也不等同于用真值证明了每个像素的 depth/normal 预测精度。

断电、进程中断或暂时下载失败后，重新执行原命令即可。不要删除 work_dir、换另一个 work_dir 重跑半成品，
也不要同时启动下载器/标注器/训练器修改数据。备份和已移出的内容在 work_dir，处理完成后仍保留。

独立重新检查已经完成的全量数据：

```bash
/absolute/path/recam_refine_work/runtime/env/bin/python -m recam_refine check \
  /absolute/path/recam_lerobot \
  --report-dir /absolute/path/recam_recheck \
  --workers 4
```

只读统计 PointWorld 重合情况：

```bash
/absolute/path/recam_refine_work/runtime/env/bin/python -m recam_refine overlap \
  --episode-manifest /absolute/path/episode_manifest.jsonl \
  --pointworld-cameras /absolute/path/pointworld/droid/cameras \
  --report /absolute/path/pointworld_overlap.json
```

验证记录见 [VALIDATION.md](VALIDATION.md)。PointWorld 方法的来源为 [data 分支](https://github.com/NVlabs/PointWorld/tree/data)，
相机发布数据来自 [PointWorld-DROID](https://huggingface.co/datasets/nvidia/PointWorld-DROID)。
本目录优化目标改编代码保留 Apache-2.0 声明，许可证见 [POINTWORLD_LICENSE](POINTWORLD_LICENSE)。
