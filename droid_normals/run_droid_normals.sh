#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
DATASET_NAME="${DROID_NORMALS_DATASET_NAME:-recam_lerobot}"
DATASET_DIR="$PROJECT_ROOT/DATA/$DATASET_NAME"
CAMERAS="${DROID_NORMALS_CAMERAS:-01,02}"
MAX_ATTEMPTS="${DROID_NORMALS_MAX_ATTEMPTS:-3}"

usage() {
  echo "Usage:"
  echo "  bash $0 [--miniforge-home PATH] download <chunks> [repo_id]"
  echo "  bash $0 [--miniforge-home PATH] install <exp_name>"
  echo "  bash $0 [--miniforge-home PATH] check <exp_name> [gpu_id]"
  echo "  bash $0 [--miniforge-home PATH] convert <chunks> <exp_name> [gpu_ids]"
  echo
  echo "Default dataset: ./DATA/$DATASET_NAME (override with DROID_NORMALS_DATASET_NAME)."
  echo "Default cameras/retries: $CAMERAS / $MAX_ATTEMPTS."
}

check_name() {
  [[ "$1" =~ ^[A-Za-z0-9._-]+$ ]] || {
    echo "ERROR: invalid name: $1" >&2
    exit 2
  }
}

if [[ "${1:-}" == "--miniforge-home" ]]; then
  [[ $# -ge 3 ]] || { usage; exit 2; }
  export MINIFORGE_HOME="$2"
  shift 2
fi
if [[ -z "${MINIFORGE_HOME:-}" ]]; then
  echo "ERROR: pass --miniforge-home /xxxx/miniforge/xxxx or set MINIFORGE_HOME." >&2
  exit 2
fi
if [[ ! -x "$MINIFORGE_HOME/bin/conda" ]]; then
  echo "ERROR: conda does not exist at $MINIFORGE_HOME/bin/conda" >&2
  exit 1
fi
export PATH="$MINIFORGE_HOME/bin:$PATH"
check_name "$DATASET_NAME"

case "${1:-}" in
  download)
    [[ $# -ge 2 && $# -le 3 ]] || { usage; exit 2; }
    REPO_ID="${3:-${DROID_NORMALS_REPO_ID:-}}"
    if [[ -z "$REPO_ID" ]]; then
      echo "ERROR: pass repo_id or set DROID_NORMALS_REPO_ID." >&2
      exit 2
    fi
    exec bash "$SCRIPT_DIR/download_droid_rgb_inputs.sh" \
      "$2" "$DATASET_DIR" "$REPO_ID" "$CAMERAS"
    ;;

  install)
    [[ $# -eq 2 ]] || { usage; exit 2; }
    check_name "$2"
    exec bash "$SCRIPT_DIR/install_normalcrafter.sh" "$PROJECT_ROOT/Res/$2"
    ;;

  check)
    [[ $# -ge 2 && $# -le 3 ]] || { usage; exit 2; }
    check_name "$2"
    export DROID_NORMALS_CHECK_ONLY=1
    exec bash "$SCRIPT_DIR/run_droid_normals_conversion.sh" \
      "0" "$PROJECT_ROOT/DATA/.check" "${3:-0}" "$PROJECT_ROOT/Res/$2" "01" "1"
    ;;

  convert)
    [[ $# -ge 3 && $# -le 4 ]] || { usage; exit 2; }
    check_name "$3"
    exec bash "$SCRIPT_DIR/run_droid_normals_conversion.sh" \
      "$2" "$DATASET_DIR" "${4:-all}" "$PROJECT_ROOT/Res/$3" \
      "$CAMERAS" "$MAX_ATTEMPTS"
    ;;

  *)
    usage
    exit 2
    ;;
esac
