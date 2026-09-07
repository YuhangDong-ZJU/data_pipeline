# 验证记录（2026-09-06）

## 2026-09-07：匹配 PyTorch 2.8.0+cu129

- 按对方环境锁定 GPU `torch==2.8.0+cu129`，CPU 检查环境同步为 `torch==2.8.0+cpu`。
  两份含 SHA-256 的锁文件由官方 PyTorch wheel 索引重新解析生成，其他核心依赖继续使用原固定版本。
- 在独立目录全新安装 GPU/CPU 环境。GPU 环境在屏蔽 GPU 时完成准备和 CPU 数值检查；
  实际 CUDA 检查在 RTX 4090 / NVIDIA 580.65.06 上通过。官方 wheel 报告的架构列表包含 H100 使用的 `sm_90`。
- CPU 环境 40 项测试通过；GPU 环境 40 项测试通过，涵盖真实 CUDA 的目标/梯度对照、CUDA Graph、
  连续优化与断点恢复、批量独立性。CPU 完整检查流程、共享 GPU 环境两并发 worker 和运行期间锁检查通过。
- 对真实 episode 0、9 的四路外部相机，以相同 Parquet/PNG 摘要、初值、内参、采样帧和预算重新拟合。
  四个最终外参矩阵相对先前 2.5.1+cu124 测量的最大元素差均为 **0**，候选验证 loss 和接受结果也完全一致。
  三路运行 2,000 次，一路从 2,000 续到总计 6,000 次。所有读取的原始样例摘要在前后相同。
  这是小样本版本回归，不能推断所有 episode 或所有硬件版本结果相同；候选验证 loss 也不替代最终独立帧质量审计。
- 新增安装入口检查，拒绝覆盖已有不同版本的 GPU Torch；临时伪造旧版安装 metadata 的 CLI 测试证明
  拒绝发生在包下载/安装之前，原 metadata 保持不变。旧环境和已开始的作业应保留原版本恢复。
- Debian 12 CI 采用新 CPU 锁文件，并覆盖全部分步、共享环境和版本保护入口。

产物位于独立验证目录 `torch28_benchmark_20260907/`，包括逐相机候选、输入摘要和 `version_comparison.json`。
没有在对方 H100 节点实跑；没有修改既有环境或原始 DROID/simulation 样例。

## 2026-09-07：两个 GPU worker 共用一套环境

- `shard-refine --runtime-work-dir` 显式选择预装共享环境；不修改分片算法协议，原工作目录和优化断点继续可用。
  日志、驱动/编译缓存和 Adam 状态保留在各自 worker 目录，旧 worker 环境不会自动覆盖或删除。
- 40 项测试通过，包括真实 POSIX 共享读锁、安装独占锁冲突和软链接别名保护。
- 新增真实 bootstrap CLI 子进程测试：第一个子进程仍运行时，第二个读者可以验证并运行；
  安装器无法取得独占锁；子进程退出后独占锁恢复可用。
- 两个并发 Bash worker 共用同一套 CPU 测试环境，验证日志/缓存隔离、旧 runtime 文件保持、
  未准备环境及与数据/协调/worker 目录冲突的环境路径拒绝，随后合并、写回成功。原独立环境 Bash 入口也通过。
- 在同一服务器的两张 RTX 4090 上，两个并发 worker 共用同一套 CUDA 环境，实际执行 GPU doctor 中的
  grid_sample 前后向及批量优化 CUDA Graph 检查，并完成合成分片的合并和写回。
  此入口样例的外参全部来自模拟发布结果，没有待优化 episode；不是全量吞吐或新一轮外参质量测量。
- CI 增加 Debian 12 中的共享环境并发入口和完整子进程生命周期锁检查。

所有数据均为临时构造，环境位于独立验证目录；未改写原始真实/仿真样例。
未进行两台物理 H100 或对方共享文件系统实测。共享目录仍须满足 [MULTI_MACHINE.md](MULTI_MACHINE.md) 的文件锁要求。

## 2026-09-07：CPU/GPU 阶段分离

- 新建独立环境安装 `torch==2.5.1+cpu` 和带 SHA-256 的完整锁定依赖，屏蔽全部 GPU。
  37 项测试全部通过，包含批量优化数学对照；未改动优化目标、分片协议或质量门槛。
- 实际 Bash 分步流程额外执行 `check --prepared-runtime`：完整解码 DROID 2 个、simulation 1 个
  合成 episode，然后在 CPU 上完成两个 DROID episode 的真实几何指标计算。
  合成的常量深度与机器人几何不匹配，正确产生 `QUALITY_REVIEW_REQUIRED.json` 并拒绝 cleanup；
  没有把测试样例伪装成质量合格的数据。检查前后训练文件内容摘要一致。
- 两分片 Bash 测试通过：CPU reference 测试路径使用预装环境，保持独立 worker 日志、完整覆盖与显式合并/写回。
- 在无可见 GPU 的主机进程中执行 `--prepare-gpu`；另在没有 NVIDIA 设备挂载的 Debian 12 / glibc 2.36
  用户空间中验证已准备的 CUDA wheel 环境能执行 CPU 数值检查。CPU-only 环境在该 Debian 12 中运行 37 项测试全部通过。
- 实际 RTX 4090 使用 `--gpu --verify-only`，将 HTTP/HTTPS 代理设为不可达地址后仍通过锁定版本检查、
  `grid_sample` 前后向与批量优化 CUDA Graph 检查。没有网络安装阶段。
- CI 改为安装 CPU-only Torch，并运行上述包含真实几何测量和质量拒绝门槛的 Bash smoke。

上述测试只使用隔离环境、临时数据和测试目录。未改写原始 DROID/simulation 样例。
这验证了环境和执行路径；没有在对方两台物理 H100 机器或其共享文件系统上实跑，
也没有证明其平台的 GPU 空闲回收策略一定不会触发。输入读取和断点 I/O 仍可能短暂不使用 GPU。
执行说明见 [CPU_GPU_STAGES.md](CPU_GPU_STAGES.md)。

## 更新：按功能独立执行

新增 `run_step.sh`，顺序为 transfer → unpack → align → overlap → refine → apply → check → cleanup。
每步单独停止，候选生成不写回数据，检查通过后也不会自动整理。

本次验证：

- Debian 12 / glibc 2.36 用户空间中 **21 项测试全部通过**。测试代码和独立运行时绑定进用户命名空间，未修改宿主系统。
- 新增完整分步测试从真实格式的 1280×720、uint16 PNG 迁移开始，覆盖旧 TAR 与新深度冲突保护、
  已裁过的数据与后补 normal 再对齐、对齐写入中断后重跑、PointWorld 发布矩阵转换、候选不写回、
  显式写回及 wrist/状态保持、全量媒体检查、检查失败/检查后数据改变时拒绝整理、最终 TAR/日志策略。
- 另测第一步在没有 normal、RGB、PointWorld 依赖的数据目录上独立完成；第二个相机的 PNG 损坏时，
  在移动任何数据文件之前停止。修复输入后使用原命令恢复，重复执行不重复迁移。
- 同盘 `/tmp` 与跨盘 `/tmp` → `/data2/recam_refine_validation_20260906` 分别实测移动/复制和重跑，均通过。
- 直接执行交付的 Bash 入口及 CLI：transfer、unpack、align、overlap、refine、apply、status；
  未通过 check 的 cleanup 被拒绝，TAR 和日志仍在。
- GPU 入口验证自动环境检查，在 8 张 RTX 4090 上完成 PyTorch 2.5.1 + CUDA 12.4 的实际 `grid_sample` 前向/反向；
  没有对方 H100 机器的直接访问权限，因此不声称已在那台 8×H100 上实跑。
- Bash 语法与 Ruff 检查通过；CI 增加 Debian 12 的 Bash/CLI 命令测试。CPU-only CI 的两个 Torch 依赖测试会跳过，
  上述带 Torch 的 Debian 12 用户空间验证则运行了全部 21 项。

分步测试使用临时构造的 DROID + simulation 数据，未改写 `/data2/droid` 或 `/data/dyh/recam_lerobot`。
**分步调度测试用替身控制几何验收结果，以覆盖允许/拒绝整理两条路径；不把合成样例当作真实外参质量证据。**
几何指标自身另有独立测试，真实样例的质量、目标函数/FK 对照证据仍见下文；这次拆分没有更改优化目标或质量门槛。
也没有将 18k+ episode 全量重新跑一遍，不能由这些测试保证所有真实 episode 都会通过几何验收。

复现命令（在仓库根目录，运行时已经由 bootstrap 建立）：

```bash
"$RECAM_WORK/runtime/env/bin/python" -m unittest discover -s recam_refine/tests -v
"$RECAM_WORK/runtime/env/bin/python" -m recam_refine.tests.smoke_manual_entry \
  --runtime-work-dir "$RECAM_WORK"
```

有空闲 CUDA GPU 时为第二条命令加 `--gpu-bootstrap`，可一并检查 GPU 环境入口。
日志保存在测试目录 `manual_debian12_validation.log`、`manual_cli_validation.log`、`manual_cli_gpu_validation.log`。

## 更新：双相机质量审计与可视化

此前的 7.72 → 6.08 cm 来自 episode 9 的 **external_1 单相机、少量验证帧**，
不是整个 episode 的外参质量结论，也不是位姿真值误差。新增审计发现其 external_2 在默认 2,000 次时尚未收敛：
训练 0.09587 m、选择验证帧 0.11990 m，未通过门槛，旧逻辑保留了初值。
在同一 DROID 初值、同一 16 个采样时刻和同一划分上重跑 6,000 次后，训练 0.06215 m、选择验证 0.07256 m，
原门槛不变且通过。此次重跑约 526 秒。脚本现会对可观测但默认优化失败的相机自动执行这一有界重试。

最终评估另选 **24 个未用于拟合或候选选择的帧**，两个外部相机一起评估。所有候选使用相同 ReCam FS 深度和内参。

| Episode | 外参 | 深度 L1，按点数加权 (cm) | F1@5 mm (%) | F1@20 mm (%) |
| --- | --- | ---: | ---: | ---: |
| 0 | DROID 初值 | 17.004 | 1.058 | 41.870 |
| 0 | 本脚本试拟合 | 7.048 | 51.542 | 68.715 |
| 0 | PointWorld 发布外参 | 7.001 | 58.893 | 69.013 |
| 9 | DROID 初值 | 16.595 | 18.497 | 57.476 |
| 9 | 本脚本，external_2 自动延长预算 | 5.893 | 63.863 | 74.739 |

Episode 0 的本地试拟合仍比 PointWorld 在 F1@5 mm 上低约 7.35 个百分点；不能声称本地优化已全面达到作者质量。
实际生产流程对这个重合 episode 使用 PointWorld 发布外参。Episode 9 没有对应发布外参，无法做该 episode 的直接三方比较。
对全部候选采用固定机器人排除 mask 并集后，episode 9 的 F1@5/20 mm 仍从 20.19/60.96% 提升到 64.64/74.91%。
这是额外的 mask 敏感性检查，不混入主表。样本很少，不代表 18,155 条全量质量。

生成了每个样例三个预先确定时刻、两个相机的 RGB/URDF 轮廓叠图，以及统一比例的双色点云俯视图，
另有指标柱状图、按同一可用 episode 配对的累积曲线、HTML、PNG、PDF 和逐帧 JSON。
服务器产物位于 `/data2/recam_refine_validation_20260906/camera_final/`。
全部读取的源 Parquet、评估深度 PNG、RGB MP4 在读前/读后 SHA-256 相同。

从 PointWorld 源文件 AST 直接执行原版目标，与移植目标在三个真实时刻对照：loss/有效点数相同；
6-DoF 梯度在 float32 容差内一致。另用原版 urdfpy 比较 27 个 link 的 FK，三个时刻的最大矩阵元素误差 < 6e-16。
这些验证用的 urdfpy/lxml 装在独立试验目录，不属于交付运行依赖。

扩展后的 **15 项测试**在原服务器和 Debian 12 / glibc 2.36 用户空间均通过。
新增覆盖 F1 的方向计数/时间聚合、已知位姿回投、三角形近裁剪、评估帧隔离、残差小但 F1 退步的拒绝诊断、
有界重试、只读报告生成/输入摘要复用，以及几何检查前延后 TAR/日志清理。
实际样例也通过了多进程审计。详细口径及复现差异见 [CAMERA_QUALITY.md](CAMERA_QUALITY.md)。

下面保留此前的数据恢复、媒体和环境验证记录。

实现与命令均已运行验证。验证没有改写用户提供的 `/data2/droid` 或 `/data/dyh/recam_lerobot` 数据；
写入仅发生在独立的 `/data2/recam_refine_validation_20260906` 和临时测试目录中。

## 真实数据与 PointWorld 重合

使用服务器原始 `episode_manifest.jsonl` 的 18,155 条记录，并独立下载 PointWorld 官方 camera 包重新统计：

| 结果 | episode 数 |
| --- | ---: |
| UUID、两个外部相机 serial、发布成功状态、合法矩阵、final loss < 0.10 均通过 | 12,225 |
| 发布包中没有对应 episode | 5,930 |
| 合计 | 18,155 |

本次 PointWorld Hub revision：`dd9aaeec94bb14e27ab6b16b6e4aa0dbcf3ef56f`。
本次 DROID SVO manifest Hub revision：`6538fa8da933b07a88d8e365eb1b383748a18a03`。
官方相机包解出了 42,935 个 JSON。也验证了自动下载 chunk-000 的 1,000 条映射。

## 外参优化实测

使用真实 Parquet、FoundationStereo 深度和 PointWorld 官方机器人网格，针对 external_1 运行 2,000 次优化。
从 10 个均匀采样时刻过滤可见性，交替分配训练/独立验证帧。以下误差为网格投影深度 L1，**不是外参真值误差**。

| 原 ReCam episode | 独立验证误差：初始 → 优化后 | 位姿平移变化 | 旋转变化 | 验收 |
| --- | --- | ---: | ---: | --- |
| 0 | 0.05917 → 0.05548 m | 0.01034 m | 1.412° | 通过 |
| 9（PointWorld 发布包未覆盖） | 0.07725 → 0.06077 m | 0.03043 m | 2.307° | 通过 |

这两个实验在 RTX 4090 上运行，分别约 109 和 126 秒。生产流程每 episode 最多采样 16 帧，并处理两个外部相机；
此处耗时不能直接视为生产全量或 H100 的耗时。所有读取的原 Parquet / 深度文件在实验前后核对 SHA-256 一致。

## 真实媒体

- 对真实 DROID external_1 RGB、实际 NormalCrafter 输出、LIBERO RGB 和 LIBERO normal 各裁出 12 帧；
  解码后的 YUV 像素与各自原视频前缀逐帧完全相同，PTS/FPS/帧数通过。
- 核对上述 4 个原视频处理前后的 SHA-256 一致。
- 读取并检查 DROID 两个外部相机共 24 张真实深度 PNG。
- 从 LIBERO 原 TAR 中只读提取 12 张 PNG 到测试目录，验证 PNG 位深和 CRC；原 TAR 保留且未改写。

## 独立运行环境

在独立目录从零安装固定版本 Python、CPU/GPU wheel，并执行带 hash 校验的 lock 文件安装。
PyAV/libx264、NumPy、PyArrow 和 PyTorch CUDA 的实际算子检查均通过；8 张可见 RTX 4090 的
`grid_sample` 前向/反向检查均通过。

另从 Docker 官方 Debian bookworm-slim 镜像建立独立的 **Debian 12 / glibc 2.36** 用户空间，
用无特权 user/mount/pid namespace 运行同一 Python 环境和测试。镜像 digest：
`sha256:5ae3c39ebd15e229dcedd5cee596b2497182493d41ff162e824ba13fc1b2b867`。
Debian 12 环境中的 doctor 和 8 项测试全部通过。没有使用 Docker daemon，也没有修改系统环境。

目标 H100 机器未提供连接，因此**没有声称在对方那台 8×H100 上运行过**。
交付的 CUDA 12.4 PyTorch wheel 支持 H100；安装器会在实际机器上执行逐卡运行检查。

## 自动化覆盖

`python -m unittest discover -s recam_refine/tests -v`：8 项测试通过，覆盖：

- TAR 路径穿越/错误相机目录拒绝、已有 PNG 冲突时不覆盖。
- 带 B 帧的视频精确裁剪、解码像素前缀相同。
- PointWorld 求逆、物理 serial 匹配、反射矩阵拒绝。
- 可微深度投影目标的实际优化和独立验证帧验收。
- 旧 episode 已裁短、后补 normal 尚未裁短、另一个 episode 有 2 帧全零 padding 的混合全流程。
- Parquet 和全局索引、统计值、wrist/动作/状态摘要、元数据、simulation 保留 TAR、DROID 删除 TAR、日志移出。
- 完成后重跑不修改数据；模拟中断后从原备份重建，不重复裁剪。
- 同盘原子移动、跨 `/tmp` 和 `/data2` 文件系统复制、转移恢复。

Ruff 检查通过。GitHub Actions 增加了 Debian 12 的独立安装和数据处理测试。
这些是环境、算法样例和恢复机制的验证，**尚未在对方完整数据上执行全量处理/逐帧检查**；
完整数据运行后的结论以其 work_dir 的 `SUCCESS.json`、各子集检查报告和外参保留报告为准。

## 双视角融合对比（2026-09-06）

新增只读 `visualize-fusion`。在 codex-218 的原始样例上实际导出：

- Episode 9，第 173 帧：DROID 初值与本地 PointWorld 方法优化结果；两台相机分别保留 45,177 / 65,195 个点。
- Episode 0，第 288 帧：DROID 初值与 PointWorld 发布外参。
- 均为此前 24 帧独立审计的中间代表帧；不是拟合/候选选择帧。两组比较的源像素完全相同。
- 输出原色/按相机着色的 PLY、PNG/PDF、源 RGB、可同步旋转缩放的离线 HTML 和输入/输出 SHA-256。
- Episode 9 的共享桌面裁剪范围是 `[0.36,-0.38,-0.16]` 至 `[0.70,0.18,0.08]` 米。
  图中可见桌面双层和部分物体横向错位减小，夹爪周围仍有残差；不宣称全区域精确重合。
- 原 PNG/MP4 先与审计 SHA-256 比对，导出后再次验证一致；没有改写源数据。
- 浏览器实际检查了双相机着色、真实 RGB 切换、左右旋转同步。
- 新增固定像素身份/边界裁剪和 PLY 几何、RGB、相机 ID 保真测试；服务器独立 runtime 和 Debian 12 隔离用户空间均通过 17 项测试，Ruff 通过。

可视化没有增加 ICP、平滑或重建步骤，也不把可视化中的含机器人点集冒充 PointWorld 去机器人掩码后的 F1 评估点集。

## Viser 查看器（2026-09-06）

- 对同一 Episode 9 / 第 173 帧导出 URDF 正运动学网格和双相机视锥；Parquet、融合 NPZ 和源 RGB 摘要均验证通过。
- 在 Windows 独立 Python 3.12 venv 中使用 Viser 1.1.0；服务仅监听 `127.0.0.1:8870`。
- 浏览器实际验证了优化前后切换、RGB → 相机双色 → RGB 恢复、相机视锥显隐及半透明机器人叠加。
- 两个独立 HTML 场景实际显示了正确的前后点云及模型，可嵌入并排页面；导出数据不依赖远端样例服务器。
- 查看器依赖采用单独的 hash lock，不修改主流程 CPU/GPU lock。Ruff 与原有 17 项测试通过。

## GPU 批量加速（2026-09-07）

- GPU 加速交付时完整测试集 **30 项通过**，分别在服务器锁定 runtime（含真实 CUDA）和 Debian 12 隔离用户空间运行。
- 新增损失/梯度和像素首样本去重对照、CUDA Graph、连续优化与恢复一致性、相机独立性、
  不可观测相机隔离、原子断点合并/拆分、2,000 次边界恢复、输入变更拒绝、动态调度和显存回退测试。
- 8 张 RTX 4090 均通过批量优化器与 CUDA Graph 的实际算子检查；无额外依赖、系统环境修改或模型下载。
- 实际 Bash/CLI 的独立步骤、只生成候选、未检查禁止 cleanup 均通过。Ruff 通过。
- 用真实 4-episode / 双 GPU 和 12-episode / 单 GPU 运行生产候选生成调度；原始输入 SHA-256 前后一致。
- 对 episode 0 和 9 的新候选重新完成每个 episode 24 帧的独立几何审计，均通过相对官方初值的门槛。
  新旧候选有小幅数值差异；完整速度、质量数值和限制见 [PERFORMANCE.md](PERFORMANCE.md)。
- 上述样例通过不等同于对方完整数据已经运行或 H100 已经实测。最终验收仍由对方完整 `check` 决定。

## 共享数据、多机分片（2026-09-07）

- 完整测试集扩至 **37 项通过**，在真实 CUDA 的服务器环境及 Debian 12 隔离用户空间分别运行。
- 分片测试覆盖：稳定不重叠的划分、缺片拒绝、并发读取/重复 worker/写操作互斥、修改输入/计划/候选拒绝、
  单机与分片模式混用拒绝、已发布候选换 worker 复用、合并中断恢复、merge 前不能 apply、check 前不能 cleanup。
- NFS 目录锁异常的兼容路径及禁用远端锁的挂载选项使用模拟测试；没有实际双主机 NFS 挂载的验证条件。
- 实际 Bash/CLI 在临时完整 fixture 上走通分片规划、两个独立 worker 目录、状态、合并和显式 apply；
  提前 merge、显式不同迭代预算、未检查 cleanup 均按预期失败。原逐步入口 smoke 和 Ruff 也通过。
- 真实样例仅复制 episode 0、9、1、2 的实际优化输入到独立验证目录，保留原始编号：
  两个 CLI 进程分别用 RTX 4090 GPU 6/7 各处理一片，合并后与同输入单机运行比较。
  4 个 episode / 8 个相机的接受状态一致，外参矩阵最大元素差均为 **0.0**。
- 该真实输入副本只具备外参拟合所需的 Parquet/采样 PNG/meta，未冒充完整可训练 LeRobot 数据集，
  未对其执行 apply/cleanup。复制后的输入及原始 `/data2/droid` 源文件 SHA-256 前后一致。
  多机完整数据写回边界由前述完整临时 fixture 验证；原样例目录不做改动。
- 两个 worker 的外层墙钟时间为 **44.18 s**，不含规划/合并、样例准备和运行环境安装。
  此结果只证明同一服务器上独立进程和共享本地目录的功能及候选一致性，不是实际两台服务器、
  网络存储或 H100 的扩展性实测；全量 ETA 与最终质量仍需目标机器实测和完整 `check`。

执行文档：[MULTI_MACHINE.md](MULTI_MACHINE.md)。工作记录保存在数据集外，分片流程只在显式 `apply` 后写回外参。
