#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${TRIM_ENV_NAME:-recam_data_pipeline}"

if [[ $# -lt 2 ]]; then
  echo "Usage:"
  echo "  bash $0 <dataset_root> <conversion_output> [more_outputs ...] [--dry-run]"
  echo
  echo "Example:"
  echo "  bash $0 /data2/droid /data2/droid_depth_output --dry-run"
  exit 2
fi

CONDA_BIN="${DROID_DEPTH_CONDA_BIN:-${DATA_PIPELINE_CONDA_BIN:-}}"
if [[ -z "$CONDA_BIN" && -n "${MINIFORGE_HOME:-}" ]]; then
  CONDA_BIN="$MINIFORGE_HOME/bin/conda"
fi
if [[ -z "$CONDA_BIN" ]]; then
  CONDA_BIN="$(command -v conda || true)"
fi
if [[ -z "$CONDA_BIN" ]]; then
  for candidate in "$HOME/miniforge3/bin" "$HOME/miniconda3/bin"; do
    if [[ -x "$candidate/conda" ]]; then
      CONDA_BIN="$candidate/conda"
      break
    fi
  done
fi
if [[ -z "$CONDA_BIN" || ! -x "$CONDA_BIN" ]]; then
  echo "ERROR: Conda or Miniforge was not found."
  echo "Set MINIFORGE_HOME, DATA_PIPELINE_CONDA_BIN or DROID_DEPTH_CONDA_BIN."
  exit 1
fi
export PATH="$(dirname -- "$CONDA_BIN"):$PATH"

if ! "$CONDA_BIN" env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  echo "Creating Conda environment: $ENV_NAME"
  "$CONDA_BIN" create \
    --name "$ENV_NAME" \
    --channel conda-forge \
    --override-channels \
    python=3.10 numpy pyarrow ffmpeg \
    --yes
fi

exec "$CONDA_BIN" run --no-capture-output --name "$ENV_NAME" \
  python "$ROOT/trim_padded_tails.py" "$@"
