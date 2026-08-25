# DROID depth utilities

Utilities for integrating and maintaining FoundationStereo depth annotations in
a LeRobot-formatted DROID dataset. These scripts do not run FoundationStereo;
they operate on depth outputs that have already been generated.

## Contents

- `move_depth_output.py` moves completed `observation.images.depth_*` episode
  directories into a target dataset on the same filesystem. Conversion
  annotations and logs remain in the source output directory.
- `pack_depth.py` packs depth PNG episode directories into uncompressed,
  episode-aligned TAR shards. Source deletion is optional and only happens after
  a shard has been written and verified.
- `trim_padded_tails.py` removes padded tail timesteps from all modalities of
  affected LeRobot v2.1 episodes, using FoundationStereo conversion metadata.
- `run_trim_padded_tails.sh` creates or reuses the required Conda environment
  and launches `trim_padded_tails.py`.

## Move completed depth outputs

Preview the move plan first:

```bash
python droid_depth/move_depth_output.py \
  /data2/droid_depth_output \
  /data2/droid \
  --dry-run
```

Remove `--dry-run` after the source, destination, episode count, and chunks have
been checked. The source and target must be separate directories on the same
filesystem.

## Pack depth PNGs

Preview TAR shards without writing them:

```bash
python droid_depth/pack_depth.py /data2/droid \
  --chunks chunk-000 chunk-001 \
  --episodes-per-shard 250 \
  --dry-run
```

Use `--delete-source` only when the verified TAR files should replace the source
episode directories.

## Trim FoundationStereo-padded tails

Preview all dataset changes:

```bash
bash droid_depth/run_trim_padded_tails.sh \
  /data2/droid \
  /data2/droid_depth_output \
  --dry-run
```

The launcher defaults to the Conda environment `recam_data_pipeline`. Set
`TRIM_ENV_NAME` to use a different name and `MINIFORGE_HOME` when Conda is not
already available in `PATH`. Remove `--dry-run` only after the reported episode
and frame counts have been reviewed.
