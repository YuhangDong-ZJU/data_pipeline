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
LONG_VIDEO_PATCH="$SCRIPT_DIR/normalcrafter_long_video.patch"
BOUNDED_INPUT_PATCH="$SCRIPT_DIR/normalcrafter_bounded_input.patch"
ENV_PROFILE="${DROID_NORMALS_ENV_PROFILE:-auto}"
ATTENTION_BACKEND="${DROID_NORMALS_ATTENTION_BACKEND:-auto}"
H100_TORCH_VERSION="${DROID_NORMALS_H100_TORCH_VERSION:-2.8.0}"
H100_XFORMERS_VERSION="${DROID_NORMALS_H100_XFORMERS_VERSION:-0.0.32.post2}"
H100_TORCH_INDEX_URL="${DROID_NORMALS_H100_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

if [[ "$ENV_PROFILE" == "auto" ]]; then
  GPU_NAMES="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true)"
  if grep -Eiq 'H100|H200|B100|B200' <<<"$GPU_NAMES"; then
    ENV_PROFILE="h100"
  else
    ENV_PROFILE="legacy"
  fi
fi
if [[ "$ENV_PROFILE" != "legacy" && "$ENV_PROFILE" != "h100" ]]; then
  echo "ERROR: DROID_NORMALS_ENV_PROFILE must be auto, legacy or h100." >&2
  exit 2
fi
if [[ "$ATTENTION_BACKEND" == "auto" ]]; then
  if [[ "$ENV_PROFILE" == "h100" ]]; then
    ATTENTION_BACKEND="pytorch"
  else
    ATTENTION_BACKEND="xformers"
  fi
fi
if [[ "$ATTENTION_BACKEND" != "pytorch" && "$ATTENTION_BACKEND" != "xformers" ]]; then
  echo "ERROR: DROID_NORMALS_ATTENTION_BACKEND must be auto, pytorch or xformers." >&2
  exit 2
fi

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

apply_runtime_patch() {
  local patch_path="$1"
  local patch_name="$2"
  if git -C "$NORMALCRAFTER_ROOT" apply --check "$patch_path" 2>/dev/null; then
    git -C "$NORMALCRAFTER_ROOT" apply "$patch_path"
  elif git -C "$NORMALCRAFTER_ROOT" apply --reverse --check "$patch_path" 2>/dev/null; then
    echo "NormalCrafter $patch_name patch is already applied."
  else
    echo "ERROR: the $patch_name patch does not match $NORMALCRAFTER_ROOT." >&2
    exit 1
  fi
}
apply_runtime_patch "$LONG_VIDEO_PATCH" "long-video"
apply_runtime_patch "$BOUNDED_INPUT_PATCH" "bounded-input"

REQUIREMENTS="$NORMALCRAFTER_ROOT/requirements.txt"
REQUIREMENTS_HASH="$(sha256sum "$REQUIREMENTS" | awk '{print $1}')"
READY_MARKER="$WORK_DIR/.normalcrafter-environment-ready"
READY_VALUE="normalcrafter-env-v2 requirements=$REQUIREMENTS_HASH profile=$ENV_PROFILE attention=$ATTENTION_BACKEND"
ENV_CHECK=(
  "$CONDA_PREFIX/bin/python" "$SCRIPT_DIR/check_normal_environment.py"
  --profile "$ENV_PROFILE" --attention-backend "$ATTENTION_BACKEND"
)

if "${ENV_CHECK[@]}" >/dev/null 2>&1; then
  echo "Reusing the compatible $ENV_NAME environment."
else
  "$CONDA_BIN" run -n "$ENV_NAME" python -m pip install --upgrade pip setuptools wheel
  if [[ "$ENV_PROFILE" == "h100" ]]; then
    if ! "$CONDA_PREFIX/bin/python" "$SCRIPT_DIR/check_normal_environment.py" \
        --profile h100 --attention-backend pytorch --torch-only >/dev/null 2>&1; then
      "$CONDA_PREFIX/bin/python" -m pip uninstall -y \
        torch triton xformers \
        nvidia-cublas-cu11 nvidia-cuda-cupti-cu11 nvidia-cuda-nvrtc-cu11 \
        nvidia-cuda-runtime-cu11 nvidia-cudnn-cu11 nvidia-cufft-cu11 \
        nvidia-curand-cu11 nvidia-cusolver-cu11 nvidia-cusparse-cu11 \
        nvidia-nccl-cu11 nvidia-nvtx-cu11 || true
      "$CONDA_PREFIX/bin/python" -m pip install \
        "torch==$H100_TORCH_VERSION" --index-url "$H100_TORCH_INDEX_URL"
    fi
    MODERN_REQUIREMENTS="$WORK_DIR/requirements-h100.txt"
    grep -Eiv \
      '^(torch|triton|xformers|nvidia-[A-Za-z0-9_-]+-cu11)([<=>[:space:]]|$)' \
      "$REQUIREMENTS" > "$MODERN_REQUIREMENTS"
    "$CONDA_PREFIX/bin/python" -m pip install -r "$MODERN_REQUIREMENTS"
    if [[ "$ATTENTION_BACKEND" == "xformers" ]]; then
      "$CONDA_PREFIX/bin/python" -m pip install "xformers==$H100_XFORMERS_VERSION"
    fi
  else
    "$CONDA_PREFIX/bin/python" -m pip install -r "$REQUIREMENTS"
  fi
  "$CONDA_BIN" run -n "$ENV_NAME" python -m pip install --upgrade hf_xet
fi

if ! "${ENV_CHECK[@]}"; then
  echo "ERROR: NormalCrafter environment is incompatible: $CONDA_PREFIX" >&2
  exit 1
fi
printf '%s\n' "$READY_VALUE" > "$READY_MARKER"

mkdir -p "$WORK_DIR/hf_cache"
echo "NormalCrafter environment ready."
echo "  Conda environment: $ENV_NAME ($CONDA_PREFIX)"
echo "  Environment:       $ENV_PROFILE"
echo "  Attention:         $ATTENTION_BACKEND"
echo "  Source:            $NORMALCRAFTER_ROOT"
echo "  Checkpoint cache:  $WORK_DIR/hf_cache"
