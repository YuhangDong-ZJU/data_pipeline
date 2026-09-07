# ReCam 全量恢复与 DROID refine

逐步执行入口：`recam_refine/run_step.sh`。目标环境为 **Debian 12、Linux x86_64、8 × H100**。
无需 sudo、Conda、Docker、nvcc、ZED SDK、FoundationStereo/NormalCrafter 模型权重。
脚本直接使用已经生成的深度 PNG、normal MP4 和 Parquet 中的机器人状态。

## 推荐：每次只执行一个功能

已有 clone 的更新命令、每一步的命令和完成标志，见 **[STEP_BY_STEP.md](STEP_BY_STEP.md)**。

| 顺序 | `run_step.sh` 的第一个参数 | 本次执行范围 |
| --- | --- | --- |
| 1 | `transfer` | 迁移 chunk 2–13 的 external_1/2 metric depth，逐 PNG 校验，然后停止 |
| 2 | `unpack` | 解包所有子集的 depth TAR；新迁移深度优先；暂时保留 TAR |
| 3 | `align` | DROID 各模态共同裁尾，重建 metadata/stats、修正来源；保留初始外参 |
| 4a | `overlap` | 统计 PointWorld UUID/serial 重合，输出逐 episode 明细 |
| 4b | `refine` | 复用发布外参、优化其余 episode；只生成候选，不写回数据 |
| 4c | `apply` | 显式写回上一步外参候选，并同步 metadata/stats |
| 5a | `check` | 全部媒体完整解码、metadata 检查、全 episode 独立帧几何审计 |
| 5b | `cleanup` | 确认检查后数据未变，再删除 DROID TAR/跨盘源副本、移出日志等辅助内容 |

每步都有独立日志和完成标志；中断后使用相同命令和 work_dir 恢复。
`status` 查看完成状态。没有前一步完成记录时拒绝跳步；不会自动运行下一步。
第一步只安装 CPU 环境，不依赖 normal 或 PointWorld。
可以在 4b 后使用本文后面的审计与点云命令先看候选，再选择何时执行 4c。

新 CUDA refine 使用批量 FP32 / CUDA Graph、每卡常驻进程、动态队列和单批预读。
默认自动按空闲显存选择批量，可失败续跑到总计 6,000 次并保存 Adam 状态。
使用方式见逐步文档的 4b，速度与质量验证见 **[PERFORMANCE.md](PERFORMANCE.md)**。

两台机器共享数据时，4b 可改为固定分片、多机并行和主机合并：
`shard-plan → shard-refine → shard-merge`。两边共享同一协调目录，各自使用独立运行环境和断点目录；
后续仍手动 `apply → check → cleanup`。完整命令见 **[MULTI_MACHINE.md](MULTI_MACHINE.md)**。

## 自动模式兼容入口（使用独立 work_dir）

下面是保留的整套自动执行方式。按上述逐步流程操作时，使用 `run_step.sh`，
不要把同一个 work_dir 交给 `run_refine.sh`；脚本也会拒绝混用两种流程。

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

入口脚本最后会自动执行全量外参几何审计：每个 episode 抽取 8 个未用于本次优化/选择的帧，
计算两个外部相机的机器人深度残差和双视角 F1@5/20 mm，4 个 CPU 进程并行。
几何检查通过后才最终删除 DROID TAR、清理跨盘源 PNG、移出辅助日志。
全量扫描不生成逐 episode 大图，避免产生数十 GB 的预览；选定样例的大图用下方命令生成。
**发现指标退步、几何不可用或深度残差超过门槛时，入口返回状态码 2，并保留详细报告；这表示质量待复核，不是环境安装失败。**
此时数据已对齐，但仍处于离线处理中；保留 `READY_FOR_GEOMETRY_AUDIT.json`，不会生成最终 `SUCCESS.json`。

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

入口最后新增的几何审计见 `camera_audit/index.html`、`camera_audit/summary.json`：
`camera_audit/COMPLETE.json` 只表示所有要求的测量已完成。
若存在 `camera_audit/QUALITY_REVIEW_REQUIRED.json`，仍有相机/episode 需要复核，不能仅凭前面的 `SUCCESS.json`
宣布外参质量已全部达标。审计不会再次改写数据；修正候选后可以单独重跑审计。

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

## 可视化比较与 PointWorld 质量指标

在仓库根目录运行。使用同一个 work_dir，脚本从不可变 plan 取出处理前外参，避免错误地把已更新 Parquet 当成初值：

```bash
/absolute/path/recam_refine_work/runtime/env/bin/python -m recam_refine audit-cameras \
  /absolute/path/recam_lerobot \
  --work-dir /absolute/path/recam_refine_work \
  --report-dir /absolute/path/recam_refine_work/camera_gallery \
  --episodes 0 3 9 --frames 24 --device cuda:0
```

打开 `camera_gallery/index.html` 查看相同 RGB 上的 URDF 轮廓、双色点云和指标图。
每个 episode 的 `metrics.json` 保存实际帧号、相机矩阵、各帧点数、统计口径和输入 SHA-256。
图像时刻固定选取评估序列的开头、中间、末尾，不根据改善幅度挑图。PNG/PDF 可直接导出。
中断重跑会校验报告的输入和图片摘要，复用有效结果；输入或设置变化则重新计算。

重新执行全量审计（不重新优化）：

```bash
/absolute/path/recam_refine_work/runtime/env/bin/python -m recam_refine audit-cameras \
  /absolute/path/recam_lerobot \
  --work-dir /absolute/path/recam_refine_work \
  --report-dir /absolute/path/recam_refine_work/camera_audit \
  --episodes all --frames 8 --image-frames 0 --workers 4 --device cpu --fail-on-review
```

只有原始样例、尚未执行 run 时，可提供 `--episode-manifest`、`--pointworld-cameras`，加上 `--fit`。
这会在 report_dir 试运行两个外部相机的优化、生成比较图，不写回任何源 Parquet/PNG/MP4。
缺少来源映射、深度或候选时明确报告缺失，不静默省略。`--depth-metadata` 可提供 FS sidecar 内参。
同一 report_dir 中的试拟合有输入/设置摘要，输入改变时要求换新报告目录，防止误用旧候选。

质量口径详见 [CAMERA_QUALITY.md](CAMERA_QUALITY.md)。**6 cm 的机器人网格深度残差不等于外参平移误差 6 cm**；
PointWorld 发布代码的 0.10 m 筛选线也不是高精度操作的误差保证。

### 双视角点云融合：优化前后并排旋转

已有相机审计后，可从任一评估帧生成离线 3D 页面、PNG/PDF 和带 `camera_id` 的 PLY：

```bash
/absolute/path/recam_refine_work/runtime/env/bin/python -m recam_refine visualize-fusion \
  /absolute/path/recam_lerobot \
  --metrics /absolute/path/recam_refine_work/camera_gallery/episode_000009/metrics.json \
  --output-dir /absolute/path/recam_refine_work/fusion_episode_000009
```

打开输出的 `index.html`，拖动/缩放会同步左右视角，可切换相机双色和真实 RGB。
默认取该 episode 的中间代表帧；`--frame` 可指定审计中的其他评估帧。
`--after pointworld_release` 可查看 PointWorld 发布外参，`--after recam_candidate` 查看本地优化候选。
两者都有时默认展示本地候选；实际生产数据的最终外参来源应以审计记录为准。
`--detail-bounds XMIN YMIN ZMIN XMAX YMAX ZMAX` 额外生成指定区域的俯视/侧视放大图，坐标单位为米，原点为机器人基座。

前后使用相同源深度像素和内参，只更换外参。工作空间裁剪按前后成员的并集固定，
不会靠改变保留点集掩盖偏差；保留机械臂，不做 ICP、平滑或补面。
固定图的显示范围相同，PLY 保留全部选中的点。
输入 PNG/MP4 必须与审计记录的 SHA-256 相符，导出后再次核对源文件未改变。
输出必须位于数据集外，并与审计目录分开。HTML 内嵌 Plotly，不需要网络或额外查看环境。
换帧、外参版本或局部裁剪时使用新的输出目录，避免混用旧图。

### Viser：半透明机器人与相机 RGB 视锥

若希望像 PointWorld 项目页一样，直接检查点云与机器人模型是否贴合，可把上述融合结果打包：

```bash
/absolute/path/recam_refine_work/runtime/env/bin/python -m recam_refine prepare-viewer \
  /absolute/path/recam_lerobot \
  --fusion-dir /absolute/path/recam_refine_work/fusion_episode_000009 \
  --metrics /absolute/path/recam_refine_work/camera_gallery/episode_000009/metrics.json \
  --robot-urdf /absolute/path/recam_refine_work/pointworld/assets/franka_description/franka_panda_robotiq_2f85_og.urdf \
  --output-dir /absolute/path/recam_refine_work/viser_episode_000009
```

机器人网格由该帧关节/夹爪状态做正运动学得到；切换前后时模型与观察视角固定，只改变点云和相机的外参。
打包时重新核对 Parquet 与审计记录的摘要，输出放在数据集外。查看器只读取这个可搬移的包。

Viser 是可选环境，不改变生产处理环境。在查看用的计算机上建立独立 venv：

```bash
python3 -m venv /absolute/path/recam_viewer_env
/absolute/path/recam_viewer_env/bin/python -m pip install --require-hashes -r recam_refine/requirements-viewer.lock
/absolute/path/recam_viewer_env/bin/python recam_refine/viser_viewer.py \
  /absolute/path/recam_refine_work/viser_episode_000009 \
  --port 8870 --export-dir /absolute/path/recam_refine_work/viser_episode_000009/static
```

打开 `http://127.0.0.1:8870`。可切换优化前后、真实 RGB/相机双色、点大小、模型不透明度，以及相机/坐标轴显隐。
服务只监听本机，不上传场景。导出的 `before.html` / `after.html` 是独立可旋转的 3D 页面，内嵌数据与查看器，
无需 Python 服务即可打开；独立页面不包含实时 Python 控件。
