# 验证记录（2026-09-06）

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
