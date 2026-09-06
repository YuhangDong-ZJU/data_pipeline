# 验证记录（2026-09-06）

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
