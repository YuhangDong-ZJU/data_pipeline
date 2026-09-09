#!/usr/bin/env bash
# Implementation shared by the four repository-root entry scripts.
set -Eeuo pipefail
GROUP="${1:-}"
shift || true
case "$GROUP" in prepare|gpu1|gpu2|finish) ;; *) echo 'Invalid machine stage' >&2; exit 2 ;; esac
DRY_RUN=0
case "${1:-}" in
  --help|-h)
    echo 'Usage: bash run_prepare.sh | run_gpu1.sh | run_gpu2.sh | run_cpu_finish.sh [--dry-run]'
    echo 'Order: CPU prepare -> GPU1 + GPU2 in parallel -> CPU finish. Failed groups resume on rerun.'
    echo 'Shared paths/defaults: recam_refine/launchers/config.sh; environment overrides are supported.'
    exit 0 ;;
  --dry-run) DRY_RUN=1; shift ;;
esac
[[ $# == 0 ]] || { echo 'Unknown arguments; use --help' >&2; exit 2; }
export REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"
source "$REPO_DIR/recam_refine/launchers/config.sh"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
unset PYTHONHOME PYTHONPATH
STATE=(python3 recam_refine/launchers/state.py)
STARTED=$SECONDS
CURRENT="initialization"
trap 'rc=$?; printf "[%s] FAILED elapsed=%ss stage=%s exit_code=%s\n" "$GROUP" "$((SECONDS-STARTED))" "$CURRENT" "$rc" >&2; exit "$rc"' ERR
run() {
  printf '+ '; printf '%q ' "$@"; printf '\n'
  if [[ "$DRY_RUN" == 0 ]]; then "$@"; fi
}
done_marker() {
  [[ "$DRY_RUN" == 0 ]] || return 1
  local rc=0
  "${STATE[@]}" marker "$1" || rc=$?
  case "$rc" in 0) return 0 ;; 3) return 1 ;; *) exit "$rc" ;; esac
}
step() {
  local name="$1" marker="$2"
  shift 2
  CURRENT="$name"
  if done_marker "$marker"; then
    printf '[%s] 已完成，跳过：%s\n' "$GROUP" "$name"
  else
    run bash recam_refine/run_step.sh "$name" "$RECAM_ROOT" "$RECAM_WORK" \
      --python "$SHARED_PYTHON" --no-install "$@"
  fi
}
if [[ "$DRY_RUN" == 0 ]]; then
  "${STATE[@]}" paths
  command -v flock >/dev/null || { echo 'ERROR: flock is required (Debian util-linux)' >&2; exit 1; }
  mkdir -p "$RECAM_WORK/launchers"
  # Install tee before acquiring the lease, so the logger cannot retain the lock.
  exec > >(tee -a "$RECAM_WORK/launchers/run_${GROUP}.log") 2>&1
  exec 9>"$RECAM_WORK/launchers/workflow.lock"
  case "$GROUP" in
    gpu1|gpu2) flock --shared --nonblock 9 ;;
    *) flock --exclusive --nonblock 9 ;;
  esac
fi
printf '[%s] RUNNING elapsed=0s\nREPO_DIR=%s\nRECAM_ROOT=%s\nRECAM_WORK=%s\n' \
  "$GROUP" "$REPO_DIR" "$RECAM_ROOT" "$RECAM_WORK"

if [[ "$GROUP" == prepare ]]; then
  # A completed preparation is immutable while either GPU is running.
  if done_marker launchers/PREPARE_READY.json; then
    "${STATE[@]}" ready >/dev/null
    echo 'PREPARE already complete; run GPU1 and GPU2. No environment or data changes.'
    exit 0
  fi
  CURRENT='prepare shared environment'
  # Select an existing CUDA 2.8.0+cu129 environment on CPU, then add only any
  # missing CPU-check dependencies to that SAME interpreter. No GPU required.
  run python3 -m recam_refine.environment "$RECAM_WORK" --profile prepare-gpu
  SHARED_PYTHON='<prepared-shared-python>'
  if [[ "$DRY_RUN" == 0 ]]; then SHARED_PYTHON="$("${STATE[@]}" python prepare-gpu)"; fi
  unset RECAM_REFINE_ENV_NAME
  export RECAM_REFINE_PYTHON="$SHARED_PYTHON"
  run python3 -m recam_refine.environment "$RECAM_WORK" --profile cpu --python "$SHARED_PYTHON"

  if ! done_marker exclude_episode_006795/SUCCESS.json; then
    CURRENT='timestamp scan'
    SCAN_REPORT="$RECAM_WORK/depth_timestamp_scan_2_13_$(date +%Y%m%d_%H%M%S)_$$.json"
    scan_rc=3
    if [[ "$DRY_RUN" == 0 ]]; then
      SCAN_REPORT="$("${STATE[@]}" scan-report)" && scan_rc=0 || scan_rc=$?
      if [[ "$scan_rc" != 0 && "$scan_rc" != 3 ]]; then exit "$scan_rc"; fi
      if [[ "$scan_rc" == 3 ]]; then
        SCAN_REPORT="$RECAM_WORK/depth_timestamp_scan_2_13_$(date +%Y%m%d_%H%M%S)_$$.json"
      fi
    fi
    if [[ "$scan_rc" == 3 ]]; then
      run python3 recam_refine/scan_depth_timestamps.py --depth-output "$DEPTH_OUTPUT" \
        --chunks 2-13 --report "$SCAN_REPORT"
      if [[ "$DRY_RUN" == 0 ]]; then SCAN_REPORT="$("${STATE[@]}" scan-report)"; fi
    fi
    step exclude-6795 exclude_episode_006795/SUCCESS.json --scan-report "$SCAN_REPORT"
  fi
  step transfer STEP1_DEPTH_TRANSFER_SUCCESS.json --depth-output "$DEPTH_OUTPUT" --depth-chunks 2-13
  step unpack STEP2_UNPACK_SUCCESS.json
  step align STEP3_ALIGN_SUCCESS.json --workers "$ALIGN_WORKERS"
  step overlap STEP4_OVERLAP_SUCCESS.json
  step shard-plan SHARD_PLAN_READY.json --num-shards 2 --refine-backend batched
  CURRENT='save preparation record'
  run "${STATE[@]}" save-ready
  echo 'NEXT: start run_gpu1.sh on GPU machine 1 and run_gpu2.sh on GPU machine 2.'
else
  CURRENT='verify preparation'
  SHARED_PYTHON='<prepared-shared-python>'
  if [[ "$DRY_RUN" == 0 ]]; then SHARED_PYTHON="$("${STATE[@]}" ready)"; fi
  unset RECAM_REFINE_ENV_NAME
  export RECAM_REFINE_PYTHON="$SHARED_PYTHON"
  case "$GROUP" in
    gpu1|gpu2)
      shard=0; worker="$WORKER_A"
      if [[ "$GROUP" == gpu2 ]]; then shard=1; worker="$WORKER_B"; fi
      CURRENT="shard-refine $shard"
      # Existing per-shard locks prevent duplicate workers. The GPU profile
      # verifies actual CUDA availability without installing or updating packages.
      run bash recam_refine/run_step.sh shard-refine "$RECAM_ROOT" "$RECAM_WORK" \
        --python "$SHARED_PYTHON" --no-install --shard-id "$shard" --worker-work-dir "$worker" \
        --devices "$GPU_DEVICES" --gpu-batch-size "$GPU_BATCH_SIZE"
      echo 'NEXT: after BOTH GPU scripts succeed, run run_cpu_finish.sh on CPU.' ;;
    finish)
      step shard-merge STEP4_REFINE_SUCCESS.json
      step apply STEP4_APPLY_SUCCESS.json --workers "$CHECK_WORKERS"
      # Do not repeat a completed full check. Cleanup still checks the stored
      # report and training-file signature before changing anything.
      step check STEP5_CHECK_SUCCESS.json --workers "$CHECK_WORKERS" --audit-frames "$AUDIT_FRAMES"
      step cleanup SUCCESS.json
      step repack REPACK_SUCCESS.json --episodes-per-shard "$EPISODES_PER_TAR" --workers "$REPACK_WORKERS"
      echo 'FINAL: DROID depth PNG + verified TAR; simulation PNG + original TAR; logs/backups outside dataset.' ;;
  esac
fi
printf '[%s] SUCCESS elapsed=%ss\n' "$GROUP" "$((SECONDS-STARTED))"
