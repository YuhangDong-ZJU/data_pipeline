#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
DATASET_NAME="${DROID_NORMALS_DATASET_NAME:-recam_lerobot}"
CAMERAS="${DROID_NORMALS_CAMERAS:-01,02}"
MAX_ATTEMPTS="${DROID_NORMALS_MAX_ATTEMPTS:-3}"
WORKSPACE_ROOT="${DROID_NORMALS_WORKSPACE_ROOT:-$PROJECT_ROOT}"

usage() {
  echo "Usage:"
  echo "  bash $0 [--workspace-root PATH] [--miniforge-home PATH] download <chunks> [repo_id]"
  echo "  bash $0 [--workspace-root PATH] [--miniforge-home PATH] install"
  echo "  bash $0 [--workspace-root PATH] [--miniforge-home PATH] check [gpu_id]"
  echo "  bash $0 [--workspace-root PATH] [--miniforge-home PATH] convert <chunks> <exp_name> [gpu_ids]"
  echo
  echo "Default workspace: repository root (override with --workspace-root or"
  echo "DROID_NORMALS_WORKSPACE_ROOT)."
  echo "Dataset: <workspace>/DATA/$DATASET_NAME (override with DROID_NORMALS_DATASET_DIR)."
  echo "Resources: <workspace>/Res (override with DROID_NORMALS_RES_DIR)."
  echo "Default cameras/retries: $CAMERAS / $MAX_ATTEMPTS."
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workspace-root)
      [[ $# -ge 2 && -n "$2" ]] || { usage; exit 2; }
      WORKSPACE_ROOT="$2"
      shift 2
      ;;
    --miniforge-home)
      [[ $# -ge 2 && -n "$2" ]] || { usage; exit 2; }
      export MINIFORGE_HOME="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      break
      ;;
  esac
done

if [[ ! -d "$WORKSPACE_ROOT" ]]; then
  echo "ERROR: workspace root does not exist: $WORKSPACE_ROOT" >&2
  exit 1
fi
WORKSPACE_ROOT="$(cd -- "$WORKSPACE_ROOT" && pwd -P)"
DATASET_DIR="$(realpath -m -- "${DROID_NORMALS_DATASET_DIR:-$WORKSPACE_ROOT/DATA/$DATASET_NAME}")"
RES_DIR="$(realpath -m -- "${DROID_NORMALS_RES_DIR:-$WORKSPACE_ROOT/Res}")"
RUNTIME_DIR="$(realpath -m -- "${DROID_NORMALS_RUNTIME_DIR:-$RES_DIR/runtime/normalcrafter}")"
EXPERIMENTS_DIR="$(realpath -m -- "${DROID_NORMALS_EXPERIMENTS_DIR:-$RES_DIR/experiments}")"

check_name() {
  [[ "$1" =~ ^[A-Za-z0-9._-]+$ ]] || {
    echo "ERROR: invalid name: $1" >&2
    exit 2
  }
}

runtime_ready() {
  local runtime_dir="$1"
  local expected_commit="75af9887a2cb14cd1ce3883c5773bc296565777c"
  local actual_commit
  actual_commit="$(git -C "$runtime_dir/NormalCrafter" rev-parse HEAD 2>/dev/null || true)"
  [[ -d "$runtime_dir/NormalCrafter/.git" \
    && "$actual_commit" == "$expected_commit" \
    && -s "$runtime_dir/.normalcrafter-environment-ready" \
    && -s "$runtime_dir/.normalcrafter-model-ready" \
    && -d "$runtime_dir/hf_cache/hub/models--Yanrui95--NormalCrafter" \
    && -d "$runtime_dir/hf_cache/hub/models--stabilityai--stable-video-diffusion-img2vid-xt" ]]
}

link_runtime() {
  local source_dir="$1"
  local target_dir="$2"
  local link_target
  mkdir -p "$(dirname -- "$target_dir")"
  link_target="$(realpath --relative-to="$(dirname -- "$target_dir")" "$source_dir")"
  ln -s "$link_target" "$target_dir"
  echo "Reused checked runtime: $target_dir -> $link_target"
}

prepare_shared_runtime() {
  local source_dir="${DROID_NORMALS_RUNTIME_SOURCE:-}"
  local candidate
  local -a candidates=()

  if runtime_ready "$RUNTIME_DIR"; then
    return 0
  fi
  if [[ -e "$RUNTIME_DIR" || -L "$RUNTIME_DIR" ]]; then
    echo "Shared runtime exists but is incomplete: $RUNTIME_DIR" >&2
    return 0
  fi

  if [[ -n "$source_dir" ]]; then
    if ! runtime_ready "$source_dir"; then
      echo "ERROR: DROID_NORMALS_RUNTIME_SOURCE is incomplete: $source_dir" >&2
      exit 1
    fi
    link_runtime "$source_dir" "$RUNTIME_DIR"
    return 0
  fi

  shopt -s nullglob
  for candidate in "$RES_DIR"/*; do
    [[ "$candidate" == "$RES_DIR/runtime" \
      || "$candidate" == "$RES_DIR/experiments" ]] && continue
    if runtime_ready "$candidate"; then
      candidates+=("$candidate")
    fi
  done
  shopt -u nullglob

  if [[ ${#candidates[@]} -eq 1 ]]; then
    link_runtime "${candidates[0]}" "$RUNTIME_DIR"
  elif [[ ${#candidates[@]} -gt 1 ]]; then
    echo "ERROR: multiple checked legacy runtimes were found:" >&2
    printf '  %s\n' "${candidates[@]}" >&2
    echo "Set DROID_NORMALS_RUNTIME_SOURCE to the one that should be reused." >&2
    exit 1
  fi
}

CONDA_BIN="${DROID_NORMALS_CONDA_BIN:-${DATA_PIPELINE_CONDA_BIN:-}}"
if [[ -z "$CONDA_BIN" && -n "${MINIFORGE_HOME:-}" ]]; then
  CONDA_BIN="$MINIFORGE_HOME/bin/conda"
fi
if [[ -z "$CONDA_BIN" ]]; then
  CONDA_BIN="$(command -v conda || true)"
fi
if [[ -z "$CONDA_BIN" || ! -x "$CONDA_BIN" ]]; then
  echo "ERROR: conda is unavailable. Use --miniforge-home, MINIFORGE_HOME," >&2
  echo "       DATA_PIPELINE_CONDA_BIN or DROID_NORMALS_CONDA_BIN." >&2
  exit 1
fi
export DROID_NORMALS_CONDA_BIN="$CONDA_BIN"
export PATH="$(dirname -- "$CONDA_BIN"):$PATH"
check_name "$DATASET_NAME"
echo "Workspace root:       $WORKSPACE_ROOT"
echo "Dataset directory:    $DATASET_DIR"
echo "Resource directory:   $RES_DIR"
echo "NormalCrafter runtime: $RUNTIME_DIR"

case "${1:-}" in
  download)
    [[ $# -ge 2 && $# -le 3 ]] || { usage; exit 2; }
    REPO_ID="${3:-${DROID_NORMALS_REPO_ID:-Sponbebob4258/recam-lerobot}}"
    DATASET_PREFIX="${DROID_NORMALS_DATASET_PREFIX-real_world/droid}"
    [[ -n "$DATASET_PREFIX" ]] || {
      echo "ERROR: DROID_NORMALS_DATASET_PREFIX cannot be empty for dataset download." >&2
      exit 2
    }
    IFS=',' read -r -a DOWNLOAD_CAMERA_ARRAY <<<"$CAMERAS"
    DOWNLOAD_MODALITIES="meta"
    for CAMERA in "${DOWNLOAD_CAMERA_ARRAY[@]}"; do
      case "$CAMERA" in
        observation.images.rgb_*) MODALITY="$CAMERA" ;;
        rgb_*) MODALITY="$CAMERA" ;;
        *[!0-9]*|'')
          echo "ERROR: invalid RGB camera for download: $CAMERA" >&2
          exit 2
          ;;
        *) MODALITY="rgb_$(printf '%02d' "$((10#$CAMERA))")" ;;
      esac
      DOWNLOAD_MODALITIES+=",$MODALITY"
    done
    DOWNLOAD_MINIFORGE_HOME="${MINIFORGE_HOME:-$(
      cd -- "$(dirname -- "$CONDA_BIN")/.." && pwd -P
    )}"
    exec bash "$PROJECT_ROOT/dataset_download/download_recam_lerobot.sh" \
      --miniforge-home "$DOWNLOAD_MINIFORGE_HOME" \
      "$DATASET_PREFIX" \
      --repo-id "$REPO_ID" \
      --destination "$DATASET_DIR" \
      --modalities "$DOWNLOAD_MODALITIES" \
      --chunks "$2"
    ;;

  install)
    [[ $# -eq 1 ]] || { usage; exit 2; }
    prepare_shared_runtime
    exec bash "$SCRIPT_DIR/install_normalcrafter.sh" "$RUNTIME_DIR"
    ;;

  check)
    [[ $# -ge 1 && $# -le 2 ]] || { usage; exit 2; }
    prepare_shared_runtime
    export DROID_NORMALS_CHECK_ONLY=1
    exec bash "$SCRIPT_DIR/run_droid_normals_conversion.sh" \
      "0" "$WORKSPACE_ROOT/DATA/.check" "${2:-0}" "$RUNTIME_DIR" \
      "$EXPERIMENTS_DIR/.check" "01" "1"
    ;;

  convert)
    [[ $# -ge 3 && $# -le 4 ]] || { usage; exit 2; }
    check_name "$3"
    prepare_shared_runtime
    RUN_DIR="$EXPERIMENTS_DIR/$3"
    exec bash "$SCRIPT_DIR/run_droid_normals_conversion.sh" \
      "$2" "$DATASET_DIR" "${4:-all}" "$RUNTIME_DIR" "$RUN_DIR" \
      "$CAMERAS" "$MAX_ATTEMPTS"
    ;;

  *)
    usage
    exit 2
    ;;
esac
