#!/usr/bin/env bash
set -Eeuo pipefail
export PYTHONNOUSERSITE=1

if [[ $# -lt 5 || $# -gt 7 ]]; then
  echo "Usage: bash $0 <chunks> <dataset_dir> <gpu_ids> <runtime_dir> <run_dir> [cameras] [max_attempts]"
  echo "Example: bash $0 0-3 ./DATA/droid all ./Res/runtime/normalcrafter ./Res/experiments/nc-v1 01,02 3"
  exit 2
fi

CHUNKS="$1"
DATASET_DIR="$2"
GPU_IDS="$3"
RUNTIME_DIR="$4"
RUN_DIR="$5"
CAMERAS="${6:-01,02}"
SUBSETS="${DROID_NORMALS_SUBSETS-real_world/droid}"
EPISODES="${DROID_NORMALS_EPISODES:-}"
MAX_ATTEMPTS="${7:-3}"
CHECK_ONLY="${DROID_NORMALS_CHECK_ONLY:-0}"
DRY_RUN="${DROID_NORMALS_DRY_RUN:-0}"
VERBOSE_INFERENCE="${DROID_NORMALS_VERBOSE_INFERENCE:-0}"
PREFETCH_NEXT_VIDEO="${DROID_NORMALS_PREFETCH_NEXT_VIDEO:-0}"
VIDEO_DECODE_BATCH_SIZE="${DROID_NORMALS_VIDEO_DECODE_BATCH_SIZE:-16}"
ATTENTION_BACKEND="${DROID_NORMALS_ATTENTION_BACKEND:-auto}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${DROID_NORMALS_ENV_NAME:-droid_normals}"
WORKER="$SCRIPT_DIR/annotate_normals_normalcrafter.py"
NORMALCRAFTER_ROOT="${NORMALCRAFTER_ROOT:-$RUNTIME_DIR/NormalCrafter}"
HF_HOME="$RUNTIME_DIR/hf_cache"
LOG_DIR="$RUN_DIR/logs"
MODEL_READY_MARKER="$RUNTIME_DIR/.normalcrafter-model-ready"
NORMALCRAFTER_COMMIT="75af9887a2cb14cd1ce3883c5773bc296565777c"
UNET_PATH="${DROID_NORMALS_UNET_PATH:-Yanrui95/NormalCrafter}"
UNET_REVISION="${DROID_NORMALS_UNET_REVISION:-7e24d68d86ae008fe08ef50b4e51cd2fc2c8cf57}"
PRETRAIN_PATH="${DROID_NORMALS_PRETRAIN_PATH:-stabilityai/stable-video-diffusion-img2vid-xt}"
PRETRAIN_REVISION="${DROID_NORMALS_PRETRAIN_REVISION:-9e43909513c6714f1bc78bcb44d96e733cd242aa}"
MODEL_READY_VALUE="$NORMALCRAFTER_COMMIT $UNET_PATH@$UNET_REVISION $PRETRAIN_PATH@$PRETRAIN_REVISION"

if [[ ! "$MAX_ATTEMPTS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: max_attempts must be a positive integer: $MAX_ATTEMPTS" >&2
  exit 2
fi
if [[ "$CHECK_ONLY" != "0" && "$CHECK_ONLY" != "1" \
    || "$DRY_RUN" != "0" && "$DRY_RUN" != "1" \
    || "$VERBOSE_INFERENCE" != "0" && "$VERBOSE_INFERENCE" != "1" \
    || "$PREFETCH_NEXT_VIDEO" != "0" && "$PREFETCH_NEXT_VIDEO" != "1" ]]; then
  echo "ERROR: check-only, dry-run, verbose-inference and prefetch settings must be 0 or 1." >&2
  exit 2
fi
if [[ ! "$VIDEO_DECODE_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: DROID_NORMALS_VIDEO_DECODE_BATCH_SIZE must be a positive integer." >&2
  exit 2
fi
if [[ "$ATTENTION_BACKEND" == "auto" ]]; then
  GPU_NAMES="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true)"
  if grep -Eiq 'H100|H200|B100|B200' <<<"$GPU_NAMES"; then
    ATTENTION_BACKEND="pytorch"
  else
    ATTENTION_BACKEND="xformers"
  fi
fi
if [[ "$ATTENTION_BACKEND" != "pytorch" && "$ATTENTION_BACKEND" != "xformers" ]]; then
  echo "ERROR: DROID_NORMALS_ATTENTION_BACKEND must be auto, pytorch or xformers." >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi is unavailable." >&2
  exit 1
fi
if [[ ! -f "$WORKER" ]]; then
  echo "ERROR: annotation worker is missing: $WORKER" >&2
  exit 1
fi

if [[ "${DROID_NORMALS_SKIP_INSTALL:-0}" != "1" ]]; then
  CONDA_BIN="${DROID_NORMALS_CONDA_BIN:-${DATA_PIPELINE_CONDA_BIN:-}}"
  if [[ -z "$CONDA_BIN" && -n "${MINIFORGE_HOME:-}" ]]; then
    CONDA_BIN="$MINIFORGE_HOME/bin/conda"
  fi
  if [[ -z "$CONDA_BIN" ]]; then
    CONDA_BIN="$(command -v conda || true)"
  fi
  if [[ -z "$CONDA_BIN" || ! -x "$CONDA_BIN" ]]; then
    echo "ERROR: conda is unavailable. Set MINIFORGE_HOME or DROID_NORMALS_CONDA_BIN." >&2
    exit 1
  fi
  export DROID_NORMALS_CONDA_BIN="$CONDA_BIN"
  export PATH="$(dirname -- "$CONDA_BIN"):$PATH"
  bash "$SCRIPT_DIR/install_normalcrafter.sh" "$RUNTIME_DIR"
  CONDA_PREFIX="$(
    "$CONDA_BIN" run -n "$ENV_NAME" python -c 'import sys; print(sys.prefix)' \
      | awk 'NF { value=$0 } END { print value }'
  )"
  if [[ -z "$CONDA_PREFIX" || ! -x "$CONDA_PREFIX/bin/python" ]]; then
    echo "ERROR: cannot resolve the Python prefix for Conda environment $ENV_NAME." >&2
    exit 1
  fi
  PYTHON="${NORMALCRAFTER_PYTHON:-$CONDA_PREFIX/bin/python}"
  export PATH="$CONDA_PREFIX/bin:$PATH"
else
  PYTHON="${NORMALCRAFTER_PYTHON:-}"
  if [[ -z "$PYTHON" ]]; then
    echo "ERROR: DROID_NORMALS_SKIP_INSTALL=1 requires NORMALCRAFTER_PYTHON." >&2
    exit 1
  fi
  export PATH="$(dirname -- "$PYTHON"):$PATH"
fi
export HF_HOME HF_HUB_CACHE="$HF_HOME/hub" HF_XET_HIGH_PERFORMANCE=1
export OMP_NUM_THREADS="${DROID_NORMALS_OMP_NUM_THREADS:-4}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
mkdir -p "$HF_HOME"

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: NormalCrafter Python is missing: $PYTHON" >&2
  exit 1
fi
if [[ ! -f "$NORMALCRAFTER_ROOT/normalcrafter/normal_crafter_ppl.py" ]]; then
  echo "ERROR: NormalCrafter checkout is missing: $NORMALCRAFTER_ROOT" >&2
  exit 1
fi
if ! command -v setsid >/dev/null 2>&1; then
  echo "ERROR: setsid is required for reliable worker shutdown." >&2
  exit 1
fi

PIDS=()
terminate_worker_groups() {
  local pid
  local any_running=0
  for pid in "${PIDS[@]}"; do
    if kill -0 -- "-$pid" 2>/dev/null; then
      kill -TERM -- "-$pid" 2>/dev/null || true
      any_running=1
    fi
  done
  if [[ "$any_running" -eq 0 ]]; then
    PIDS=()
    return
  fi
  echo "Stopping NormalCrafter worker groups..." >&2
  for _ in {1..15}; do
    any_running=0
    for pid in "${PIDS[@]}"; do
      if kill -0 -- "-$pid" 2>/dev/null; then
        any_running=1
        break
      fi
    done
    if [[ "$any_running" -eq 0 ]]; then
      break
    fi
    sleep 1
  done
  for pid in "${PIDS[@]}"; do
    if kill -0 -- "-$pid" 2>/dev/null; then
      kill -KILL -- "-$pid" 2>/dev/null || true
    fi
  done
  for pid in "${PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
  PIDS=()
}

handle_shutdown() {
  local status="$1"
  trap - INT TERM EXIT
  terminate_worker_groups
  exit "$status"
}

cleanup_on_exit() {
  local status=$?
  trap - EXIT
  terminate_worker_groups
  exit "$status"
}

trap 'handle_shutdown 130' INT
trap 'handle_shutdown 143' TERM
trap cleanup_on_exit EXIT

if [[ "$GPU_IDS" == "all" ]]; then
  GPU_IDS="$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)"
fi
IFS=',' read -r -a GPU_ARRAY <<<"$GPU_IDS"
if [[ ${#GPU_ARRAY[@]} -eq 0 ]]; then
  echo "ERROR: no GPUs were selected." >&2
  exit 1
fi
for GPU_ID in "${GPU_ARRAY[@]}"; do
  if [[ ! "$GPU_ID" =~ ^[0-9]+$ ]]; then
    echo "ERROR: invalid GPU ID: $GPU_ID" >&2
    exit 2
  fi
done

VERBOSE_ARGS=()
if [[ "$VERBOSE_INFERENCE" == "1" ]]; then
  VERBOSE_ARGS+=(--verbose-inference)
fi

check_model() {
  CHECK_GPU="${GPU_ARRAY[0]}"
  echo "Loading NormalCrafter on physical GPU $CHECK_GPU."
  CUDA_VISIBLE_DEVICES="$CHECK_GPU" "$PYTHON" "$SCRIPT_DIR/check_normalcrafter.py" \
    --worker "$WORKER" \
    --normalcrafter-root "$NORMALCRAFTER_ROOT" \
    --unet-path "$UNET_PATH" \
    --unet-revision "$UNET_REVISION" \
    --pretrain-path "$PRETRAIN_PATH" \
    --pretrain-revision "$PRETRAIN_REVISION" \
    --cpu-offload "${DROID_NORMALS_CPU_OFFLOAD:-none}" \
    --attention-backend "$ATTENTION_BACKEND" \
    "${VERBOSE_ARGS[@]}"
  printf '%s\n' "$MODEL_READY_VALUE" > "$MODEL_READY_MARKER"
}

if [[ "$CHECK_ONLY" == "1" ]]; then
  check_model
  echo "Environment and checkpoints are ready below: $RUNTIME_DIR"
  exit 0
fi

mkdir -p "$LOG_DIR"
if [[ ! -d "$DATASET_DIR" ]]; then
  echo "ERROR: dataset directory does not exist: $DATASET_DIR" >&2
  exit 1
fi
if [[ "$DRY_RUN" != "1" \
    && "$(cat "$MODEL_READY_MARKER" 2>/dev/null || true)" != "$MODEL_READY_VALUE" ]]; then
  echo "No completed model preflight was found; populating the checkpoint cache once."
  check_model
fi

# The preflight above populated and validated both pinned model revisions.  GPU
# workers must use only that local cache; otherwise every process may create its
# own hf-xet network pool and inflate the Pod's host-memory footprint.
if [[ "$DRY_RUN" != "1" ]]; then
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFUSERS_OFFLINE=1
  export HF_HUB_DISABLE_XET=1
fi

IFS=',' read -r -a CAMERA_ARRAY <<<"$CAMERAS"
SUBSET_ARGS=()
if [[ -n "$SUBSETS" ]]; then
  IFS=',' read -r -a SUBSET_ARRAY <<<"$SUBSETS"
  SUBSET_ARGS+=(--subsets "${SUBSET_ARRAY[@]}")
fi
EPISODE_ARGS=()
if [[ -n "$EPISODES" ]]; then
  IFS=',' read -r -a EPISODE_ARRAY <<<"$EPISODES"
  EPISODE_ARGS+=(--episodes "${EPISODE_ARRAY[@]}")
fi
LOCAL_WORKERS="${#GPU_ARRAY[@]}"
GLOBAL_WORKERS="${DROID_NORMALS_GLOBAL_NUM_WORKERS:-$LOCAL_WORKERS}"
WORKER_OFFSET="${DROID_NORMALS_GLOBAL_WORKER_OFFSET:-0}"
WORKER_PASSES="${DROID_NORMALS_WORKER_PASSES:-2}"
if [[ ! "$GLOBAL_WORKERS" =~ ^[1-9][0-9]*$ \
    || ! "$WORKER_OFFSET" =~ ^[0-9]+$ \
    || ! "$WORKER_PASSES" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: global worker count, offset and worker passes must be valid integers." >&2
  exit 2
fi
if (( WORKER_OFFSET + LOCAL_WORKERS > GLOBAL_WORKERS )); then
  echo "ERROR: local global-worker indexes exceed DROID_NORMALS_GLOBAL_NUM_WORKERS." >&2
  exit 2
fi

echo "Dataset:          $DATASET_DIR"
echo "Chunks:           $CHUNKS"
echo "Cameras:          ${CAMERA_ARRAY[*]}"
echo "Subsets:          ${SUBSETS:-<all discovered datasets>}"
echo "Episodes:         ${EPISODES:-<all episodes in selected chunks>}"
echo "Physical GPUs:    ${GPU_ARRAY[*]}"
echo "Global shards:    $GLOBAL_WORKERS (local offset $WORKER_OFFSET)"
echo "Attempts/video:   $MAX_ATTEMPTS"
echo "Process passes:   $WORKER_PASSES"
echo "Shared runtime:   $RUNTIME_DIR"
echo "Run logs:         $LOG_DIR"
echo "Checkpoint cache: $HF_HOME"
echo "CUDA allocator:   $PYTORCH_CUDA_ALLOC_CONF"
echo "glibc arenas:      $MALLOC_ARENA_MAX"
echo "Attention:        $ATTENTION_BACKEND"

RUN_TAG="$(date +%Y%m%d-%H%M%S)-chunks-${CHUNKS//,/_}"
WORKER_MODE_ARGS=()
if [[ "$DRY_RUN" == "1" ]]; then
  WORKER_MODE_ARGS+=(--dry-run)
fi
WORKER_MODE_ARGS+=("${VERBOSE_ARGS[@]}")
WORKER_MODE_ARGS+=(--video-decode-batch-size "$VIDEO_DECODE_BATCH_SIZE")
if [[ "$PREFETCH_NEXT_VIDEO" == "1" ]]; then
  WORKER_MODE_ARGS+=(--prefetch-next-video)
fi
FINAL_FAILED=0
for (( PASS=1; PASS<=WORKER_PASSES; PASS++ )); do
  echo "Starting asynchronous GPU worker pass $PASS/$WORKER_PASSES."
  PIDS=()
  for LOCAL_INDEX in "${!GPU_ARRAY[@]}"; do
    GPU_ID="${GPU_ARRAY[$LOCAL_INDEX]}"
    SHARD_INDEX=$((WORKER_OFFSET + LOCAL_INDEX))
    LOG_FILE="$LOG_DIR/${RUN_TAG}-pass${PASS}-gpu${GPU_ID}-shard${SHARD_INDEX}.log"
    CUDA_VISIBLE_DEVICES="$GPU_ID" setsid "$PYTHON" "$WORKER" "$DATASET_DIR" \
        --normalcrafter-root "$NORMALCRAFTER_ROOT" \
        --unet-path "$UNET_PATH" \
        --unet-revision "$UNET_REVISION" \
        --pretrain-path "$PRETRAIN_PATH" \
        --pretrain-revision "$PRETRAIN_REVISION" \
        --chunks "$CHUNKS" \
        --cameras "${CAMERA_ARRAY[@]}" \
        "${SUBSET_ARGS[@]}" \
        "${EPISODE_ARGS[@]}" \
        --num-shards "$GLOBAL_WORKERS" \
        --shard-index "$SHARD_INDEX" \
        --max-attempts "$MAX_ATTEMPTS" \
        --retry-delay-seconds "${DROID_NORMALS_RETRY_DELAY_SECONDS:-5}" \
        --stale-lock-hours "${DROID_NORMALS_STALE_LOCK_HOURS:-0.25}" \
        --lock-heartbeat-seconds "${DROID_NORMALS_LOCK_HEARTBEAT_SECONDS:-30}" \
        --cpu-offload "${DROID_NORMALS_CPU_OFFLOAD:-none}" \
        --attention-backend "$ATTENTION_BACKEND" \
        --max-res "${DROID_NORMALS_MAX_RES:-1024}" \
        --output-width "${DROID_NORMALS_OUTPUT_WIDTH:-1280}" \
        --output-height "${DROID_NORMALS_OUTPUT_HEIGHT:-720}" \
        --crf "${DROID_NORMALS_CRF:-17}" \
        --ffmpeg-preset "${DROID_NORMALS_FFMPEG_PRESET:-veryfast}" \
        --continue-on-error \
        "${WORKER_MODE_ARGS[@]}" \
        > >(tee "$LOG_FILE") 2>&1 &
    PIDS+=("$!")
  done

  PASS_FAILED=0
  for PID_INDEX in "${!PIDS[@]}"; do
    PID="${PIDS[$PID_INDEX]}"
    if ! wait "$PID"; then
      PASS_FAILED=1
    fi
    unset "PIDS[$PID_INDEX]"
  done
  PIDS=()
  if [[ "$PASS_FAILED" -eq 0 ]]; then
    FINAL_FAILED=0
    break
  fi
  FINAL_FAILED=1
  if (( PASS < WORKER_PASSES )); then
    echo "One or more workers failed; completed outputs will be skipped on the next pass." >&2
  fi
done

if [[ "$FINAL_FAILED" -ne 0 ]]; then
  echo "ERROR: pending videos still failed after $WORKER_PASSES process pass(es)." >&2
  echo "Logs: $LOG_DIR" >&2
  exit 1
fi
echo "Normal annotation complete. Logs: $LOG_DIR"
