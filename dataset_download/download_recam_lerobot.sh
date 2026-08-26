#!/usr/bin/env bash
set -Eeuo pipefail
export PYTHONNOUSERSITE=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${RECAM_DOWNLOAD_ENV_NAME:-recam_download}"

usage() {
  echo "Usage:"
  echo "  bash $0 --miniforge-home PATH [download options] <subset> [subset ...]"
  echo
  echo "Examples:"
  echo "  bash $0 --miniforge-home /path/to/miniconda3 real_world/droid"
  echo "  bash $0 --miniforge-home /path/to/miniconda3 simulation/libero"
  echo "  bash $0 --miniforge-home /path/to/miniconda3 real_world/droid simulation/libero"
  echo "  bash $0 --miniforge-home /path/to/miniconda3 real_world/droid --chunks 0-3"
  echo "  bash $0 --miniforge-home /path/to/miniconda3 real_world/droid --chunks 0 --modalities data,rgb_01"
  echo "  bash $0 --miniforge-home /path/to/miniconda3 --list-subsets"
  echo
  echo "Default destination: <repository>/DATA/recam_lerobot"
}

if [[ "${1:-}" == "--miniforge-home" ]]; then
  [[ $# -ge 2 ]] || { usage; exit 2; }
  export MINIFORGE_HOME="$2"
  shift 2
fi
if [[ -z "${MINIFORGE_HOME:-}" ]]; then
  echo "ERROR: pass --miniforge-home /path/to/miniforge or set MINIFORGE_HOME." >&2
  exit 2
fi
if [[ ! -x "$MINIFORGE_HOME/bin/conda" ]]; then
  echo "ERROR: conda does not exist at $MINIFORGE_HOME/bin/conda" >&2
  exit 1
fi
export PATH="$MINIFORGE_HOME/bin:$PATH"
eval "$(conda shell.bash hook)"

if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  conda create -n "$ENV_NAME" --override-channels -c conda-forge python=3.10 pip -y
fi
if ! conda run -n "$ENV_NAME" python -c 'import huggingface_hub, hf_xet' >/dev/null 2>&1; then
  conda run -n "$ENV_NAME" python -m pip install \
    'huggingface_hub>=0.30,<2' 'hf_xet>=1.1,<2'
fi

exec conda run --no-capture-output -n "$ENV_NAME" \
  python "$SCRIPT_DIR/download_recam_lerobot.py" "$@"
