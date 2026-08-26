# DROID normals — NormalCrafter pipeline

The launcher resolves paths from its own location. The parent directory of
`droid_normals` is treated as the ReCam project root, so commands work from any
current directory. Entering the project root first remains the simplest form:

```bash
cd /path/to/ReCam
```

Commands are launched from `droid_normals/run_droid_normals.sh`. Managed paths
are relative to the resolved ReCam project root:

```text
DATA/<dataset_name>
Res/runtime/normalcrafter/NormalCrafter
Res/runtime/normalcrafter/hf_cache
Res/experiments/<exp_name>/logs
```

The production default is `DATA/recam_lerobot/real_world/droid`. The launcher
passes `real_world/droid` as the only selected subset, so other real-world or
simulation datasets below `recam_lerobot` are not annotated accidentally.

The launcher accepts cluster-specific Miniforge locations without embedding
them in the repository. It checks, in order, `DROID_NORMALS_CONDA_BIN`,
`DATA_PIPELINE_CONDA_BIN`, `MINIFORGE_HOME/bin/conda`, and finally `conda` from
`PATH`. For a Miniforge installation at `/xxxxxx/miniforge/xxxx`, use:

```bash
export MINIFORGE_HOME=/xxxxxx/miniforge/xxxx

bash droid_normals/run_droid_normals.sh \
  download 0-3 owner/recam_lerobot

bash droid_normals/run_droid_normals.sh \
  check 0

bash droid_normals/run_droid_normals.sh \
  convert 0-3 normalcrafter_v1 all
```

`--miniforge-home /xxxxxx/miniforge/xxxx` remains available as a per-command
alternative. If only the executable path is known, set
`DATA_PIPELINE_CONDA_BIN=/absolute/path/to/bin/conda` for every toolkit, or
`DROID_NORMALS_CONDA_BIN` for this pipeline only.

`download` uses Hugging Face's resumable snapshot downloader and materializes
only `real_world/droid/meta/**` and the selected DROID chunk/camera MP4s. The repository ID is explicit
because the private/filtered 18K RGB dataset cannot be inferred from its local
directory name.

`check` creates the `droid_normals` conda environment, checks out the pinned
NormalCrafter revision, applies `normalcrafter_long_video.patch`, installs its
pinned requirements, downloads both model repositories into the shared
`Res/runtime/normalcrafter/hf_cache`, and loads the model on one GPU.

`convert` starts one background process per selected physical GPU. Each process
loads one persistent model and owns a deterministic shard. On the first run, a
single-GPU preflight populates and validates the shared checkpoint cache before
the workers start, avoiding concurrent first-download races. A restart first
filters outputs having both an atomically published MP4 and an atomically
published `status=complete` JSON, then redistributes only unfinished videos over
the available workers. Each video is attempted three times by default; if a
worker process still exits nonzero, the launcher performs a second resumable
worker pass. Logs are written per pass and GPU below
`Res/experiments/<exp_name>/logs`.
Each attempt restores the model's FP16 invariant and releases unused CUDA cache
before the next video. The launcher defaults to
`PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128` to limit long-run allocator
fragmentation. A CUDA OOM therefore remains local to one attempt instead of
leaving the persistent VAE in FP32 and poisoning all later tasks.

All experiments use the single shared runtime at `Res/runtime/normalcrafter`;
experiment directories contain logs only. For compatibility with an existing
checkout, the launcher performs a one-time controlled migration when the
standard runtime is absent. If exactly one complete legacy `Res/<name>` runtime
exists, it creates a relative symbolic link instead of copying its model files.
If several valid legacy runtimes exist, it stops and lists them; set
`DROID_NORMALS_RUNTIME_SOURCE=/path/to/ready/runtime` to select one explicitly.

Output locks contain a host, PID, Linux boot/process identity and unique owner
token. A dead owner on the same host is reclaimed immediately. For shared
multi-machine storage, live workers heartbeat every 30 seconds and an unknown
owner defaults to a 15-minute lease. Once a worker owns a recovered task, it
removes only that episode's orphan `.part.mp4` and atomic-JSON part files. Worker
processes run in separate process groups, so `Ctrl+C` terminates their Python
and FFmpeg children and lets normal lock/part cleanup run before exit.

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
`DROID_NORMALS_CRF`, `DROID_NORMALS_CPU_OFFLOAD`, `DROID_NORMALS_SUBSETS`, and
`DROID_NORMALS_DATASET_PREFIX`. Lock lease behavior can be overridden with
`DROID_NORMALS_STALE_LOCK_HOURS` and `DROID_NORMALS_LOCK_HEARTBEAT_SECONDS`.
`DROID_NORMALS_EPISODES` accepts a comma-separated episode list for bounded
end-to-end tests. Set the subset and dataset-prefix variables to empty strings only when a
standalone DROID dataset has `videos/` directly at its root.

The high-level command intentionally mirrors `droid-metric-depth`:

```text
check [gpu_id]
convert <chunks> <exp_name> [gpu_ids]
```

It defaults to dataset `recam_lerobot`, cameras `01,02`, and three attempts per
video. Override these with `DROID_NORMALS_DATASET_NAME`,
`DROID_NORMALS_CAMERAS`, and `DROID_NORMALS_MAX_ATTEMPTS`.

NormalCrafter's internal tensor-shape messages, dependency warnings, and tqdm
bars are hidden by default. Worker plans, per-video start/save timing,
retry/failure messages, and final summaries remain visible. Set
`DROID_NORMALS_VERBOSE_INFERENCE=1` only when debugging model internals.
