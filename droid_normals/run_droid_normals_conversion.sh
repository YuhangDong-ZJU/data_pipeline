#!/usr/bin/env bash
set -Eeuo pipefail
export PYTHONNOUSERSITE=1

if [[ $# -lt 4 || $# -gt 6 ]]; then
  echo "Usage: bash $0 <chunks> <dataset_dir> <gpu_ids> <work_dir> [cameras] [max_attempts]"
  echo "Example: bash $0 0-3 ./DATA/droid all ./Res/nc-v1 01,02 3"
  exit 2
fi

CHUNKS="$1"
DATASET_DIR="$2"
GPU_IDS="$3"
WORK_DIR="$4"
CAMERAS="${5:-01,02}"
SUBSETS="${DROID_NORMALS_SUBSETS-real_world/droid}"
MAX_ATTEMPTS="${6:-3}"
CHECK_ONLY="${DROID_NORMALS_CHECK_ONLY:-0}"
DRY_RUN="${DROID_NORMALS_DRY_RUN:-0}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${DROID_NORMALS_ENV_NAME:-droid_normals}"
WORKER="$SCRIPT_DIR/annotate_normals_normalcrafter.py"
NORMALCRAFTER_ROOT="${NORMALCRAFTER_ROOT:-$WORK_DIR/NormalCrafter}"
HF_HOME="$WORK_DIR/hf_cache"
LOG_DIR="$WORK_DIR/logs"
MODEL_READY_MARKER="$WORK_DIR/.normalcrafter-model-ready"

if [[ ! "$MAX_ATTEMPTS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: max_attempts must be a positive integer: $MAX_ATTEMPTS" >&2
  exit 2
fi
if [[ "$CHECK_ONLY" != "0" && "$CHECK_ONLY" != "1" \
    || "$DRY_RUN" != "0" && "$DRY_RUN" != "1" ]]; then
  echo "ERROR: DROID_NORMALS_CHECK_ONLY and DROID_NORMALS_DRY_RUN must be 0 or 1." >&2
  exit 2
fi
if [[ -n "${MINIFORGE_HOME:-}" ]]; then
  export PATH="$MINIFORGE_HOME/bin:$PATH"
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
  if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda was not found. Set MINIFORGE_HOME=/path/to/miniforge." >&2
    exit 1
  fi
  bash "$SCRIPT_DIR/install_normalcrafter.sh" "$WORK_DIR"
  eval "$(conda shell.bash hook)"
  CONDA_PREFIX="$(conda run -n "$ENV_NAME" python -c 'import sys; print(sys.prefix)')"
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
mkdir -p "$HF_HOME" "$LOG_DIR"

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: NormalCrafter Python is missing: $PYTHON" >&2
  exit 1
fi
if [[ ! -f "$NORMALCRAFTER_ROOT/normalcrafter/normal_crafter_ppl.py" ]]; then
  echo "ERROR: NormalCrafter checkout is missing: $NORMALCRAFTER_ROOT" >&2
  exit 1
fi

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

check_model() {
  CHECK_GPU="${GPU_ARRAY[0]}"
  echo "Loading NormalCrafter on physical GPU $CHECK_GPU."
  CUDA_VISIBLE_DEVICES="$CHECK_GPU" "$PYTHON" "$SCRIPT_DIR/check_normalcrafter.py" \
    --worker "$WORKER" \
    --normalcrafter-root "$NORMALCRAFTER_ROOT" \
    --cpu-offload "${DROID_NORMALS_CPU_OFFLOAD:-none}"
  printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$MODEL_READY_MARKER"
}

if [[ "$CHECK_ONLY" == "1" ]]; then
  check_model
  echo "Environment and checkpoints are ready below: $WORK_DIR"
  exit 0
fi

if [[ ! -d "$DATASET_DIR" ]]; then
  echo "ERROR: dataset directory does not exist: $DATASET_DIR" >&2
  exit 1
fi
if [[ "$DRY_RUN" != "1" && ! -f "$MODEL_READY_MARKER" ]]; then
  echo "No completed model preflight was found; populating the checkpoint cache once."
  check_model
fi

IFS=',' read -r -a CAMERA_ARRAY <<<"$CAMERAS"
SUBSET_ARGS=()
if [[ -n "$SUBSETS" ]]; then
  IFS=',' read -r -a SUBSET_ARRAY <<<"$SUBSETS"
  SUBSET_ARGS+=(--subsets "${SUBSET_ARRAY[@]}")
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
echo "Physical GPUs:    ${GPU_ARRAY[*]}"
echo "Global shards:    $GLOBAL_WORKERS (local offset $WORKER_OFFSET)"
echo "Attempts/video:   $MAX_ATTEMPTS"
echo "Process passes:   $WORKER_PASSES"
echo "Checkpoint cache: $HF_HOME"

RUN_TAG="$(date +%Y%m%d-%H%M%S)-chunks-${CHUNKS//,/_}"
WORKER_MODE_ARGS=()
if [[ "$DRY_RUN" == "1" ]]; then
  WORKER_MODE_ARGS+=(--dry-run)
fi
FINAL_FAILED=0
for (( PASS=1; PASS<=WORKER_PASSES; PASS++ )); do
  echo "Starting asynchronous GPU worker pass $PASS/$WORKER_PASSES."
  PIDS=()
  for LOCAL_INDEX in "${!GPU_ARRAY[@]}"; do
    GPU_ID="${GPU_ARRAY[$LOCAL_INDEX]}"
    SHARD_INDEX=$((WORKER_OFFSET + LOCAL_INDEX))
    LOG_FILE="$LOG_DIR/${RUN_TAG}-pass${PASS}-gpu${GPU_ID}-shard${SHARD_INDEX}.log"
    (
      set -o pipefail
      CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" "$WORKER" "$DATASET_DIR" \
        --normalcrafter-root "$NORMALCRAFTER_ROOT" \
        --chunks "$CHUNKS" \
        --cameras "${CAMERA_ARRAY[@]}" \
        "${SUBSET_ARGS[@]}" \
        --num-shards "$GLOBAL_WORKERS" \
        --shard-index "$SHARD_INDEX" \
        --max-attempts "$MAX_ATTEMPTS" \
        --retry-delay-seconds "${DROID_NORMALS_RETRY_DELAY_SECONDS:-5}" \
        --stale-lock-hours "${DROID_NORMALS_STALE_LOCK_HOURS:-24}" \
        --cpu-offload "${DROID_NORMALS_CPU_OFFLOAD:-none}" \
        --max-res "${DROID_NORMALS_MAX_RES:-1024}" \
        --output-width "${DROID_NORMALS_OUTPUT_WIDTH:-1280}" \
        --output-height "${DROID_NORMALS_OUTPUT_HEIGHT:-720}" \
        --crf "${DROID_NORMALS_CRF:-17}" \
        --ffmpeg-preset "${DROID_NORMALS_FFMPEG_PRESET:-veryfast}" \
        --continue-on-error \
        "${WORKER_MODE_ARGS[@]}" \
        2>&1 | tee "$LOG_FILE"
    ) &
    PIDS+=("$!")
  done

  PASS_FAILED=0
  for PID in "${PIDS[@]}"; do
    if ! wait "$PID"; then
      PASS_FAILED=1
    fi
  done
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
