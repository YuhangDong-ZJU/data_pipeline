#!/usr/bin/env bash
set -Eeuo pipefail
export PYTHONNOUSERSITE=1

if [[ $# -ne 1 ]]; then
  echo "Usage: bash $0 <work_dir>"
  echo "The checkout and Hugging Face cache are placed below work_dir."
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$1"
ENV_NAME="${DROID_NORMALS_ENV_NAME:-droid_normals}"
NORMALCRAFTER_ROOT="$WORK_DIR/NormalCrafter"
NORMALCRAFTER_COMMIT="75af9887a2cb14cd1ce3883c5773bc296565777c"
PATCH="$SCRIPT_DIR/normalcrafter_long_video.patch"

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
mkdir -p "$WORK_DIR"

if ! "$CONDA_BIN" env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  "$CONDA_BIN" create -n "$ENV_NAME" --override-channels -c conda-forge \
    python=3.10 pip ffmpeg git -y
fi
CONDA_PREFIX="$(
  "$CONDA_BIN" run -n "$ENV_NAME" python -c 'import sys; print(sys.prefix)' \
    | awk 'NF { value=$0 } END { print value }'
)"
if [[ -z "$CONDA_PREFIX" || ! -x "$CONDA_PREFIX/bin/python" ]]; then
  echo "ERROR: cannot resolve the Python prefix for Conda environment $ENV_NAME." >&2
  exit 1
fi
export PATH="$CONDA_PREFIX/bin:$PATH"
if [[ ! -x "$CONDA_PREFIX/bin/ffmpeg" || ! -x "$CONDA_PREFIX/bin/ffprobe" ]]; then
  "$CONDA_BIN" install -n "$ENV_NAME" --override-channels -c conda-forge ffmpeg -y
fi
for executable in git sha256sum; do
  if ! command -v "$executable" >/dev/null 2>&1; then
    echo "ERROR: $executable is required." >&2
    exit 1
  fi
done

if [[ ! -e "$NORMALCRAFTER_ROOT" ]]; then
  git clone https://github.com/Binyr/NormalCrafter.git "$NORMALCRAFTER_ROOT"
elif [[ ! -d "$NORMALCRAFTER_ROOT/.git" ]]; then
  echo "ERROR: $NORMALCRAFTER_ROOT exists but is not a Git checkout." >&2
  exit 1
fi

CURRENT_COMMIT="$(git -C "$NORMALCRAFTER_ROOT" rev-parse HEAD)"
if [[ "$CURRENT_COMMIT" != "$NORMALCRAFTER_COMMIT" ]]; then
  if ! git -C "$NORMALCRAFTER_ROOT" diff --quiet \
      || ! git -C "$NORMALCRAFTER_ROOT" diff --cached --quiet; then
    echo "ERROR: cannot change the NormalCrafter revision because it has local changes." >&2
    exit 1
  fi
  git -C "$NORMALCRAFTER_ROOT" fetch origin "$NORMALCRAFTER_COMMIT"
  git -C "$NORMALCRAFTER_ROOT" checkout --detach "$NORMALCRAFTER_COMMIT"
fi

if git -C "$NORMALCRAFTER_ROOT" apply --check "$PATCH" 2>/dev/null; then
  git -C "$NORMALCRAFTER_ROOT" apply "$PATCH"
elif git -C "$NORMALCRAFTER_ROOT" apply --reverse --check "$PATCH" 2>/dev/null; then
  echo "NormalCrafter long-video patch is already applied."
else
  echo "ERROR: the long-video patch does not match $NORMALCRAFTER_ROOT." >&2
  exit 1
fi

REQUIREMENTS="$NORMALCRAFTER_ROOT/requirements.txt"
REQUIREMENTS_HASH="$(sha256sum "$REQUIREMENTS" | awk '{print $1}')"
READY_MARKER="$WORK_DIR/.normalcrafter-environment-ready"
INSTALLED_HASH="$(cat "$READY_MARKER" 2>/dev/null || true)"
if [[ "$INSTALLED_HASH" != "$REQUIREMENTS_HASH" ]] \
    || ! "$CONDA_BIN" run -n "$ENV_NAME" python -c \
      'import torch, diffusers, transformers, accelerate, xformers, decord, cv2, hf_xet' \
      >/dev/null 2>&1; then
  "$CONDA_BIN" run -n "$ENV_NAME" python -m pip install --upgrade pip setuptools wheel
  "$CONDA_BIN" run -n "$ENV_NAME" python -m pip install -r "$REQUIREMENTS"
  "$CONDA_BIN" run -n "$ENV_NAME" python -m pip install --upgrade hf_xet
fi

if ! "$CONDA_PREFIX/bin/python" -c \
    'import torch, diffusers, transformers, accelerate, xformers, decord, cv2, hf_xet'; then
  echo "ERROR: NormalCrafter dependencies are incomplete in $CONDA_PREFIX." >&2
  exit 1
fi
printf '%s\n' "$REQUIREMENTS_HASH" > "$READY_MARKER"

mkdir -p "$WORK_DIR/hf_cache"
echo "NormalCrafter environment ready."
echo "  Conda environment: $ENV_NAME ($CONDA_PREFIX)"
echo "  Source:            $NORMALCRAFTER_ROOT"
echo "  Checkpoint cache:  $WORK_DIR/hf_cache"
