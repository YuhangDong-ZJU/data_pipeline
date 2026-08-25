#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

usage() {
  echo "Usage:"
  echo "  bash $0 [--miniforge-home PATH] download <chunks> <dataset_name> <repo_id> [cameras]"
  echo "  bash $0 [--miniforge-home PATH] install <exp_name>"
  echo "  bash $0 [--miniforge-home PATH] check <exp_name> [gpu_id]"
  echo "  bash $0 [--miniforge-home PATH] convert <chunks> <dataset_name> <exp_name> [gpu_ids] [cameras] [max_attempts]"
  echo
  echo "Paths are fixed to ./DATA/<dataset_name> and ./Res/<exp_name>."
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

case "${1:-}" in
  download)
    [[ $# -ge 4 && $# -le 5 ]] || { usage; exit 2; }
    check_name "$3"
    exec bash "$SCRIPT_DIR/download_droid_rgb_inputs.sh" \
      "$2" "$PROJECT_ROOT/DATA/$3" "$4" "${5:-01,02}"
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
    [[ $# -ge 4 && $# -le 7 ]] || { usage; exit 2; }
    check_name "$3"
    check_name "$4"
    exec bash "$SCRIPT_DIR/run_droid_normals_conversion.sh" \
      "$2" "$PROJECT_ROOT/DATA/$3" "${5:-all}" "$PROJECT_ROOT/Res/$4" \
      "${6:-01,02}" "${7:-3}"
    ;;

  *)
    usage
    exit 2
    ;;
esac
