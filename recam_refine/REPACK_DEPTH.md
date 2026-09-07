# 将最终 DROID depth PNG 重新打包

完成 `check → cleanup` 后，在 CPU 机器的仓库根目录单独执行：

```bash
bash recam_refine/run_step.sh repack "$RECAM_ROOT" "$RECAM_WORK" \
  --episodes-per-shard 250 \
  --workers 4 \
  --prepared-runtime
```

使用前面准备的 CPU 环境，不需要 GPU，也不需要额外安装 GNU tar 或压缩工具。
这是可选发布步骤，不会自动接在 `cleanup` 后运行。只处理 DROID 的 `depth_01` 和 `depth_02`。
RGB、normal、Parquet、meta、simulation 以及 wrist 数据不参与打包。

## 发布格式

沿用原 `droid_depth/pack_depth.py` 的布局，每包最多 250 个 episode，不跨 chunk 或相机。
TAR 是未压缩的 `.tar`，PNG 原始字节不变。例子：

```text
real_world/droid/images/chunk-000/observation.images.depth_01/episodes-000000-000249.tar
```

包内路径相对于 DROID 子集根目录，现有解包器可以直接恢复：

```text
images/chunk-000/observation.images.depth_01/episode_000000/frame_000000.png
```

PNG 保留在原位置，LeRobot 仍可直接训练，meta 中的 PNG 路径和帧数保持不变。
需要额外保存一份 depth 数据的磁盘空间；程序会在写包前估算新 TAR 占用并检查可用空间。
上传发布时选择 TAR 文件；当前入口不删除 PNG，也不自动上传。

## 自动校验和恢复

- 要求 `STEP5_CHECK_SUCCESS.json`、`SUCCESS.json` 内容有效，几何报告无待复核项。
- 比对最终校验记录中的训练文件状态和几何报告摘要，拒绝打包校验后发生变化的数据。
- 按最终 metadata 检查两路 depth 的 episode 和帧覆盖，拒绝额外/缺失帧、目录及未知旧 TAR。
- 写临时包时记录每张 PNG 的 SHA-256；关闭并同步临时包后，重新顺序读取每个 TAR 成员，核对路径、数量、大小及 SHA-256。
- 全部匹配后才原子发布单个 TAR；源 PNG 保留，损坏或冲突的已有 TAR 不会被覆盖。
- 中断后使用相同命令恢复；已有包仍完整读取核验，不能仅凭“文件存在”跳过。发布后尚未写回凭据的包会重新与源 PNG 比对。
- 打包结束再确认训练文件状态未改变，才生成 `REPACK_SUCCESS.json`。

文件状态校验沿用最终检查的规则：metadata 哈希，媒体大小/mtime/ctime；
新 TAR 的每张 PNG 另外逐字节计算 SHA-256，包本身也记录全文件 SHA-256。
重新核验会读取已有 TAR 的全部内容，因此恢复执行仍有磁盘 I/O。

完成提示为 `REPACK COMPLETE`。日志为 `$RECAM_WORK/step_repack.log`，
逐包 SHA-256 和成员清单在 `$RECAM_WORK/repacked_depth/`，整体记录在 `$RECAM_WORK/REPACK_SUCCESS.json`。
失败返回非零状态并记录 `REPACK_FAILED.json`，不会生成有效的整体打包完成标志。

`--workers` 是同时打包/核验的 CPU/I/O 线程数；共享盘通常先用 4。
恢复时可以调整并发数，`--episodes-per-shard` 及输入数据保持原值。

新增入口没有修改分片计划所绑定的优化协议文件。已有计划和优化结果无需重建；
仍在运行的任务结束后，再在 CPU 机器统一更新共享代码并执行本步骤。
