# Quick depth sidecar scan

Run on a CPU machine with Python 3.9 or newer. Standard library only; no bootstrap,
Conda activation, GPU, model download or PNG decoding is needed.

```bash
cd "$REPO_DIR"
python3 recam_refine/scan_depth_timestamps.py \
  --depth-output "$DEPTH_OUTPUT" \
  --report "$RECAM_WORK/depth_timestamp_scan_$(date +%Y%m%d_%H%M%S).json"
```

The input can also be the `annotations/foundation_stereo_depth` directory.
Reports must be outside the input directory; existing reports are never overwritten.
Run after conversion writers have stopped. The scan does not modify source files.

- `affected_episodes`: distinct episode filenames with any detected anomaly,
  including unreadable JSON and inconsistent metadata.
- `anomalous_camera_records`: affected per-camera JSON files.
- `duplicate_final_retry_episodes`: distinct episodes with exactly one equal
  adjacent timestamp pair at the decoded tail, one recovered retry frame and no
  declared padding (the episode 6795 pattern). This is a signature, not proof
  of a missing first image.
- `affected_episode_ids` and `anomalies` provide identifiers and exact details.

Normal trailing `null` timestamps for declared padding are allowed. A retry alone
is not an anomaly. A clean sidecar does not prove PNG/RGB alignment: a monotonic
shift cannot be detected without source-frame evidence. The report is not an
automatic deletion manifest. Removing an episode later must handle all cameras,
modalities, episode/index mappings, and dataset metadata together.
