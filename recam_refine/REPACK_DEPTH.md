# 将最终 DROID depth PNG 重新打包

完成 `check → cleanup` 后，在 CPU 机器的仓库根目录单独执行：

```bash
bash recam_refine/run_step.sh repack "$RECAM_ROOT" "$RECAM_WORK" \
  --episodes-per-shard 250 \
  --workers 4 \
  --prepared-runtime
```

使用前面准备的 CPU 环境，不需要 GPU，也不需要额外安装 GNU tar 或压缩工具。
`run_cpu_finish.sh` 会在 `cleanup` 后执行此步骤，也可以单独执行。只处理 DROID 的 `depth_01` 和 `depth_02`。
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
直接开始打包，不估算占用，也不查询剩余磁盘空间。
上传发布时选择 TAR 文件；当前入口不删除 PNG，也不自动上传。

## 自动校验和恢复

- 要求 `STEP5_CHECK_SUCCESS.json`、`SUCCESS.json` 内容有效，几何报告无待复核项。
- 比对最终校验记录中的训练文件状态和几何报告摘要，拒绝打包校验后发生变化的数据。
- 按最终 metadata 检查两路 depth 的 episode 和帧覆盖，拒绝额外/缺失帧、目录及未知旧 TAR。
- 直接写入临时 TAR，检查成员覆盖、写入长度及源文件在写入期间是否变化，然后原子发布；不计算 PNG/TAR 内容哈希，不复读新包。
- 已完成且文件属性未变化的包跳过；只有缺少完成记录的已有包才与源 PNG 比较，恢复断点。
- 打包结束确认完成记录及 TAR 文件属性未变，生成 `REPACK_SUCCESS.json`，不再遍历训练文件。

源 PNG 保留。写入失败时移除临时包，不覆盖冲突的已有 TAR。
完成提示为 `REPACK COMPLETE`。日志为 `$RECAM_WORK/step_repack.log`，
成员清单和文件属性在 `$RECAM_WORK/repacked_depth/`，整体记录在 `$RECAM_WORK/REPACK_SUCCESS.json`。
失败返回非零状态并记录 `REPACK_FAILED.json`，不会生成有效的整体打包完成标志。

`--workers` 是同时打包的 CPU/I/O 线程数；四阶段入口默认使用 8。
恢复时可以调整并发数，`--episodes-per-shard` 及输入数据保持原值。

新增入口没有修改分片计划所绑定的优化协议文件。已有计划和优化结果无需重建；
仍在运行的任务结束后，再在 CPU 机器统一更新共享代码并执行本步骤。
