# Exclude confirmed source episode 6795

CPU only. Run after the timestamp scan and **before successful transfer or unpack**.
Do not run dataset readers, downloaders, annotators or other processing concurrently.
This command is intentionally limited to the confirmed single-episode report.

```bash
# First update codex/recam-dataset-refine and restore REPO_DIR, RECAM_ROOT, RECAM_WORK.
cd "$REPO_DIR"

# Create a named fresh report (JSON-only scan, no new environment).
export SCAN_REPORT="$RECAM_WORK/depth_timestamp_scan_2_13_$(date +%Y%m%d_%H%M%S).json"
python3 recam_refine/scan_depth_timestamps.py \
  --depth-output "$DEPTH_OUTPUT" \
  --chunks 2-13 \
  --report "$SCAN_REPORT"

# Reuse an installed environment; supplement missing packages if necessary.
bash recam_refine/run_step.sh exclude-6795 "$RECAM_ROOT" "$RECAM_WORK" \
  --reuse-env \
  --scan-report "$SCAN_REPORT"

# Only after EXCLUSION COMPLETE, continue the original CPU 1 transfer command.
bash recam_refine/run_step.sh transfer "$RECAM_ROOT" "$RECAM_WORK" \
  --reuse-env \
  --depth-output "$DEPTH_OUTPUT" \
  --depth-chunks 2-13
```

The command retains file backups and a recovery plan under
`$RECAM_WORK/exclude_episode_006795/`. On interruption rerun the exact exclusion
command; other stages refuse an incomplete exclusion. A failed transfer with no
receipts is supported; its old frozen configuration is backed up and invalidated.

The original last episode (18154 in the full dataset) fills index 6795. Its
`source_episode_index` remains 18154 in `meta/episodes.jsonl`. The source manifest,
SVO sidecars, depth transfer and PointWorld matching use that original identity.
No video re-encoding or depth inference occurs. The relative order changes for
this single replacement episode; all episode indices remain contiguous.

Both affected episodes' media and sidecars are backed up. Shared depth TARs are
rewritten without the excluded or old last-episode members, retaining and checking
the other members. Last-episode depth PNGs are moved to the replacement identity;
existing PNGs take priority over older TAR versions. Subsequent unpack cannot
restore the excluded episode. External conversion outputs remain unchanged:
their excluded sidecars and PNGs are ignored through the retained catalogue.

Parquet global indices, replacement episode indices, episode statistics, aggregate
statistics, frame/image/video/episode/chunk counts and the train split are updated.
Task IDs/catalogue and calibration conventions remain stable. Unknown metadata or
nonstandard splits stop before changes instead of being silently discarded.

This is an identity/index consistency operation. Continue the normal align,
refinement and final full-data check to validate media and physical alignment.
The full dataset has 18154 episodes after exclusion. The script has no generic
automatic "delete every anomalous episode" option.
