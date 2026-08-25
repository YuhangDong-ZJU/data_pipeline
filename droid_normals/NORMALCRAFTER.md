# DROID normals — NormalCrafter pipeline

All commands are launched from `droid_normals/run_droid_normals.sh`. Its managed
paths are fixed to:

```text
droid_normals/DATA/<dataset_name>
droid_normals/Res/<exp_name>/NormalCrafter
droid_normals/Res/<exp_name>/hf_cache
droid_normals/Res/<exp_name>/logs
```

Pass the Miniforge root explicitly. It must be the directory containing
`bin/conda`, for example `/xxxxxx/miniforge/xxxx/`:

```bash
MF=/xxxxxx/miniforge/xxxx

bash droid_normals/run_droid_normals.sh --miniforge-home "$MF" \
  download 0-3 droid_18k owner/droid_18k 01,02

bash droid_normals/run_droid_normals.sh --miniforge-home "$MF" \
  check normalcrafter_v1 0

bash droid_normals/run_droid_normals.sh --miniforge-home "$MF" \
  convert 0-3 droid_18k normalcrafter_v1 all 01,02 3
```

`download` uses Hugging Face's resumable snapshot downloader and materializes
only `meta/**` and the selected chunk/camera MP4s. The repository ID is explicit
because the private/filtered 18K RGB dataset cannot be inferred from its local
directory name.

`check` creates the `droid_normals` conda environment, checks out the pinned
NormalCrafter revision, applies `normalcrafter_long_video.patch`, installs its
pinned requirements, downloads both model repositories into the experiment's
`hf_cache`, and loads the model on one GPU.

`convert` starts one background process per selected physical GPU. Each process
loads one persistent model and owns a deterministic shard. On the first run, a
single-GPU preflight populates and validates the shared checkpoint cache before
the workers start, avoiding concurrent first-download races. A restart first
filters outputs having both an atomically published MP4 and an atomically
published `status=complete` JSON, then redistributes only unfinished videos over
the available workers. Each video is attempted three times by default; if a
worker process still exits nonzero, the launcher performs a second resumable
worker pass. Logs are written per pass and GPU below `Res/<exp_name>/logs`.

The output is written beside RGB as
`videos/chunk-*/observation.images.normal_*/episode_*.mp4`. Defaults are
1280x720 H.264, CRF 17, `yuv420p`, and 15 FPS. Model inference uses a 1024x576
working resolution by default because direct 1280x720 inference exceeded 24 GB
in the tested implementation.

For several machines sharing or synchronizing one output tree, assign disjoint
global shards. For four machines with eight GPUs each, use the same conversion
command and set:

```bash
# machine 0, then use offsets 8, 16 and 24 on the other machines
export DROID_NORMALS_GLOBAL_NUM_WORKERS=32
export DROID_NORMALS_GLOBAL_WORKER_OFFSET=0
```

If machines receive disjoint chunks, no global settings are necessary. Useful
runtime overrides include `DROID_NORMALS_WORKER_PASSES`,
`DROID_NORMALS_RETRY_DELAY_SECONDS`, `DROID_NORMALS_MAX_RES`,
`DROID_NORMALS_CRF`, and `DROID_NORMALS_CPU_OFFLOAD`.
