#!/usr/bin/env bash
set -Eeuo pipefail
export PYTHONNOUSERSITE=1

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: bash $0 <chunks> <download_dir> <repo_id> [cameras]"
  echo "Example: bash $0 0-3 ./DATA/droid my-org/my-droid 01,02"
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CHUNKS="$1"
DOWNLOAD_DIR="$2"
REPO_ID="$3"
CAMERAS="${4:-01,02}"
DATASET_PREFIX="${DROID_NORMALS_DATASET_PREFIX-real_world/droid}"
ENV_NAME="${DROID_NORMALS_DOWNLOAD_ENV_NAME:-droid_normals_download}"

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

if ! "$CONDA_BIN" env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  "$CONDA_BIN" create -n "$ENV_NAME" --override-channels -c conda-forge python=3.10 pip -y
fi
if ! "$CONDA_BIN" run -n "$ENV_NAME" python -c 'import huggingface_hub, hf_xet' >/dev/null 2>&1; then
  "$CONDA_BIN" run -n "$ENV_NAME" python -m pip install --upgrade huggingface_hub hf_xet
fi

IFS=',' read -r -a CAMERA_ARRAY <<<"$CAMERAS"
export HF_XET_HIGH_PERFORMANCE=1
exec "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" \
  python "$SCRIPT_DIR/download_droid_rgb_inputs.py" \
  "$REPO_ID" "$CHUNKS" "$DOWNLOAD_DIR" \
  --cameras "${CAMERA_ARRAY[@]}" \
  --prefix "$DATASET_PREFIX" \
  --workers "${DROID_NORMALS_DOWNLOAD_WORKERS:-8}"
