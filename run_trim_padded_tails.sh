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

if [[ -n "${MINIFORGE_HOME:-}" ]]; then
  export PATH="$MINIFORGE_HOME/bin:$PATH"
fi

if ! command -v conda >/dev/null 2>&1; then
  for candidate in "$HOME/miniforge3/bin" "$HOME/miniconda3/bin"; do
    if [[ -x "$candidate/conda" ]]; then
      export PATH="$candidate:$PATH"
      break
    fi
  done
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: Conda or Miniforge was not found."
  echo "If Miniforge is installed elsewhere, run:"
  echo "  export MINIFORGE_HOME=/path/to/miniforge"
  exit 1
fi

eval "$(conda shell.bash hook)"

if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  echo "Creating Conda environment: $ENV_NAME"
  conda create \
    --name "$ENV_NAME" \
    --channel conda-forge \
    --override-channels \
    python=3.10 numpy pyarrow ffmpeg \
    --yes
fi

exec conda run --no-capture-output --name "$ENV_NAME" \
  python "$ROOT/trim_padded_tails.py" "$@"
