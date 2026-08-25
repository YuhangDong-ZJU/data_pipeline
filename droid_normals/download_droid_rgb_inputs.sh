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
ENV_NAME="${DROID_NORMALS_DOWNLOAD_ENV_NAME:-droid_normals_download}"

if [[ -n "${MINIFORGE_HOME:-}" ]]; then
  export PATH="$MINIFORGE_HOME/bin:$PATH"
fi
if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda was not found. Set MINIFORGE_HOME=/path/to/miniforge." >&2
  exit 1
fi
eval "$(conda shell.bash hook)"

if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  conda create -n "$ENV_NAME" --override-channels -c conda-forge python=3.10 pip -y
fi
if ! conda run -n "$ENV_NAME" python -c 'import huggingface_hub, hf_xet' >/dev/null 2>&1; then
  conda run -n "$ENV_NAME" python -m pip install --upgrade huggingface_hub hf_xet
fi

IFS=',' read -r -a CAMERA_ARRAY <<<"$CAMERAS"
export HF_XET_HIGH_PERFORMANCE=1
exec conda run --no-capture-output -n "$ENV_NAME" \
  python "$SCRIPT_DIR/download_droid_rgb_inputs.py" \
  "$REPO_ID" "$CHUNKS" "$DOWNLOAD_DIR" \
  --cameras "${CAMERA_ARRAY[@]}" \
  --workers "${DROID_NORMALS_DOWNLOAD_WORKERS:-8}"
