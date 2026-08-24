# ReCam Data Pipeline

Small tools for preparing the ReCam LeRobot dataset for release.

## Pack depth PNG files

`pack_depth.py` reads the following layout:

```text
recam_lerobot/
└── libero/
    └── images/
        └── chunk-000/
            └── observation.images.depth_00/
                └── episode_000000/
                    └── frame_000000.png
```

It creates uncompressed, episode-aligned TAR shards. Source files are never moved
or deleted. A TAR always contains one subset, one chunk and one camera only.

Preview the work without writing anything:

```bash
python3 pack_depth.py \
  /data/dyh/recam_lerobot \
  /path/to/recam_lerobot_depth_tar \
  --dry-run
```

Pack every subset using 250 episodes per shard:

```bash
python3 pack_depth.py \
  /data/dyh/recam_lerobot \
  /path/to/recam_lerobot_depth_tar \
  --episodes-per-shard 250 \
  --workers 1
```

Pack only selected data:

```bash
python3 pack_depth.py \
  /data/dyh/recam_lerobot \
  /path/to/recam_lerobot_depth_tar \
  --subsets libero \
  --chunks chunk-000 chunk-001 \
  --episodes-per-shard 250 \
  --workers 1
```

The output looks like this:

```text
recam_lerobot_depth_tar/
├── manifest.jsonl
└── libero/
    └── chunk-000/
        └── observation.images.depth_00/
            ├── shard-0000.tar
            ├── shard-0000.json
            ├── shard-0001.tar
            └── shard-0001.json
```

Each JSON sidecar records the included episodes, TAR size and SHA256 checksum.
Completed shards are skipped when the same command is run again. Use
`--overwrite` only when a complete shard must be rebuilt.

Archive members retain their original relative path. To restore one shard into a
subset root:

```bash
tar -xf shard-0000.tar -C /path/to/restored/libero
```

PNG files are already compressed, so the TAR shards are intentionally not gzip
compressed. The output needs approximately as much free space as the source depth
directories.
