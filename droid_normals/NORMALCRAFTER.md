# DROID normals — NormalCrafter pipeline

The launcher resolves its code from its own location and accepts an independent
ReCam workspace root. This keeps repositories, datasets and model resources as
siblings without symbolic links. For example:

```bash
cd /path/to/ReCam/data_pipeline

bash droid_normals/run_droid_normals.sh \
  --workspace-root /path/to/ReCam \
  --miniforge-home /path/to/miniforge3 \
  convert 0-3 normalcrafter_v1 all
```

Commands are launched from `droid_normals/run_droid_normals.sh`. Managed paths
are relative to the selected workspace root:

```text
/path/to/ReCam/DATA/<dataset_name>
/path/to/ReCam/Res/runtime/normalcrafter/NormalCrafter
/path/to/ReCam/Res/runtime/normalcrafter/hf_cache
/path/to/ReCam/Res/experiments/<exp_name>/logs
```

If `--workspace-root` is omitted, the repository root remains the workspace for
backward compatibility. `DROID_NORMALS_WORKSPACE_ROOT` provides the same
setting through the environment. Deployments with separate storage mounts can
override individual locations with `DROID_NORMALS_DATASET_DIR`,
`DROID_NORMALS_RES_DIR`, `DROID_NORMALS_RUNTIME_DIR`, and
`DROID_NORMALS_EXPERIMENTS_DIR`.

The production default is `DATA/recam_lerobot/real_world/droid`. The launcher
passes `real_world/droid` as the only selected subset, so other real-world or
simulation datasets below `recam_lerobot` are not annotated accidentally.

The launcher accepts cluster-specific Miniforge locations without embedding
them in the repository. It checks, in order, `DROID_NORMALS_CONDA_BIN`,
`DATA_PIPELINE_CONDA_BIN`, `MINIFORGE_HOME/bin/conda`, and finally `conda` from
`PATH`. For a Miniforge installation at `/xxxxxx/miniforge/xxxx`, use:

```bash
export MINIFORGE_HOME=/xxxxxx/miniforge/xxxx
export DROID_NORMALS_WORKSPACE_ROOT=/path/to/ReCam

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

`download` reuses `dataset_download/download_recam_lerobot.sh` and materializes
only `real_world/droid/meta/**` and the selected DROID chunk/camera MP4s. It
defaults to `Sponbebob4258/recam-lerobot`; an alternative repository ID remains
accepted as the optional final argument.

`check` creates the `droid_normals` conda environment, checks out the pinned
NormalCrafter revision, applies `normalcrafter_long_video.patch` and
`normalcrafter_bounded_input.patch`, installs the compatible runtime
requirements, downloads both model repositories at pinned revisions into the
shared `Res/runtime/normalcrafter/hf_cache`, and loads the model on one GPU.

The environment installer detects H100/H200/B100/B200 GPUs. On these machines
it keeps an existing working PyTorch 2.8+ CUDA environment (including the
partner cluster's PyTorch 2.8.0+cu129 environment), or installs the official
PyTorch 2.8 CUDA 12.8 wheel when the existing Torch cannot execute an actual
GPU kernel. The partner's CUDA 12.9 driver supports the CUDA 12.8 wheel. Other
GPUs retain the original NormalCrafter dependency profile.
Override automatic selection with `DROID_NORMALS_ENV_PROFILE=h100` or
`DROID_NORMALS_ENV_PROFILE=legacy`; an internal Torch mirror can be selected
with `DROID_NORMALS_H100_TORCH_INDEX_URL`.

PyTorch attention is selected automatically on H100-class GPUs and does not
depend on an optional xFormers CUDA extension. Existing 4090 machines retain
xFormers by default to preserve their lower-memory execution path. Set
`DROID_NORMALS_ATTENTION_BACKEND=pytorch` or `xformers` to override detection;
select xFormers on H100 only when the environment check confirms a working
xFormers kernel. The check now
runs real Torch (and, when requested, xFormers) CUDA kernels, so a wheel that
merely imports but lacks the H100 architecture is rejected before model
download or multi-GPU worker startup.

`convert` starts one supervised worker per selected physical GPU. Each worker
owns a stable shard of the full discovered task list. Completed outputs are
filtered only after sharding, so machines that start
at different times keep the same ownership. On the first run, a
single-GPU preflight populates and validates the shared checkpoint cache before
the workers start, avoiding concurrent first-download races. A restart first
filters outputs having both an atomically published MP4 and an atomically
published `status=complete` JSON. Each video is attempted three times by default.
The worker also writes an active-task marker before decode/inference/encoding.
If native code terminates the process, the supervisor retries that exact
episode-camera task up to three process launches. A third native crash, or three
caught Python failures, creates a persistent quarantine record and the GPU moves
on to its next pending task. Other GPUs are never restarted. The final command
returns nonzero when quarantined videos remain, so a partial dataset cannot be
reported as complete. Logs are written per launch and GPU below
`Res/experiments/<exp_name>/logs`; crash counters and quarantine records are
below its `recovery/` directory.

By default the worker replaces its process after every completed video. This
reloads the model but prevents Decord, FFmpeg and CUDA native state from
accumulating across a long shard. Completed MP4/JSON pairs and quarantined tasks
are skipped when the replacement process scans the shard again.
Each attempt restores the model's FP16 invariant. CUDA cache is released after
failed attempts or dtype recovery, while successful attempts reuse allocator
blocks for better throughput. The launcher defaults to
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

Each completed JSON records a configuration fingerprint, pinned model revisions,
input/output file sizes and video properties. Resume accepts compatible legacy
JSON files, while new outputs are skipped only when their fingerprint and MP4
header/size checks match the current run.

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
`DROID_NORMALS_NATIVE_CRASH_MAX_ATTEMPTS`,
`DROID_NORMALS_MAX_VIDEOS_PER_PROCESS`,
`DROID_NORMALS_RETRY_DELAY_SECONDS`, `DROID_NORMALS_MAX_RES`,
`DROID_NORMALS_CRF`, `DROID_NORMALS_CPU_OFFLOAD`, `DROID_NORMALS_SUBSETS`, and
`DROID_NORMALS_DATASET_PREFIX`. Environment overrides include
`DROID_NORMALS_ENV_PROFILE`, `DROID_NORMALS_ATTENTION_BACKEND`,
`DROID_NORMALS_H100_TORCH_VERSION`, `DROID_NORMALS_H100_XFORMERS_VERSION`, and
`DROID_NORMALS_H100_TORCH_INDEX_URL`. Lock lease behavior can be overridden with
`DROID_NORMALS_STALE_LOCK_HOURS` and `DROID_NORMALS_LOCK_HEARTBEAT_SECONDS`.
`DROID_NORMALS_EPISODES` accepts a comma-separated episode list for bounded
end-to-end tests. Set the subset and dataset-prefix variables to empty strings only when a
standalone DROID dataset has `videos/` directly at its root.

Host-memory use is bounded for multi-GPU jobs. RGB videos are decoded in
16-frame temporary batches, and the patched NormalCrafter pipeline preprocesses
and encodes CLIP/VAE inputs in `decode_chunk_size` batches. This avoids the
upstream behavior of first materializing the complete video as an FP32 tensor;
for example, 1,100 frames at 1024x576 occupy more than 7 GiB for one such RGB
tensor. Freed host heap is returned after model loading and after each episode,
and complete next-video prefetch is disabled by default. After checkpoint
preflight, workers are forced to use the local Hugging Face cache and do not
start per-process Xet download pools.

On Linux, each worker also gives the kernel `POSIX_FADV_DONTNEED` advice after
all reads of an input RGB MP4 are complete. A verified output MP4 is synchronized
and receives the same advice before its metadata is committed. Closing a file
alone does not evict Linux page-cache pages; this best-effort advice prevents a
long sequential dataset scan from charging hundreds of GiB of inactive file
cache to a memory-limited Kubernetes Pod. It never deletes or modifies an input,
and unsupported filesystems only produce one warning per worker.

These are operational settings and do not change the annotation fingerprint, so
existing valid MP4/JSON outputs remain resumable. Override the temporary decode
batch with `DROID_NORMALS_VIDEO_DECODE_BATCH_SIZE`. Complete next-video prefetch
is intentionally incompatible with active-task crash attribution. The launcher also defaults
`MALLOC_ARENA_MAX=2` to reduce retained glibc heap pages across decoder and
encoder threads.

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
