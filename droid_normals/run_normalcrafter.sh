#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
RUNTIME_DIR="${DROID_NORMALS_RUNTIME_DIR:-$PROJECT_ROOT/Res/runtime/normalcrafter}"
NORMALCRAFTER_ROOT="${NORMALCRAFTER_ROOT:-$RUNTIME_DIR/NormalCrafter}"
ENV_NAME="${DROID_NORMALS_ENV_NAME:-droid_normals}"

if [[ -z "${NORMALCRAFTER_PYTHON:-}" ]]; then
  CONDA_BIN="${DROID_NORMALS_CONDA_BIN:-${DATA_PIPELINE_CONDA_BIN:-}}"
  if [[ -z "$CONDA_BIN" && -n "${MINIFORGE_HOME:-}" ]]; then
    CONDA_BIN="$MINIFORGE_HOME/bin/conda"
  fi
  if [[ -z "$CONDA_BIN" ]]; then
    CONDA_BIN="$(command -v conda || true)"
  fi
  if [[ -n "$CONDA_BIN" && -x "$CONDA_BIN" ]]; then
    NORMALCRAFTER_PYTHON="$(
      "$CONDA_BIN" run -n "$ENV_NAME" python -c 'import sys; print(sys.executable)' \
        2>/dev/null | tail -n 1 || true
    )"
  fi
fi
NORMALCRAFTER_PYTHON="${NORMALCRAFTER_PYTHON:-}"
export HF_HOME="${HF_HOME:-$RUNTIME_DIR/hf_cache}"

if [[ ! -x "$NORMALCRAFTER_PYTHON" ]]; then
  echo "ERROR: NormalCrafter Python was not found: $NORMALCRAFTER_PYTHON" >&2
  echo "Run run_droid_normals.sh install, or set NORMALCRAFTER_PYTHON." >&2
  exit 1
fi

if [[ ! -f "$NORMALCRAFTER_ROOT/normalcrafter/normal_crafter_ppl.py" ]]; then
  echo "ERROR: NormalCrafter checkout was not found: $NORMALCRAFTER_ROOT" >&2
  echo "Run run_droid_normals.sh install, or set NORMALCRAFTER_ROOT." >&2
  exit 1
fi

exec "$NORMALCRAFTER_PYTHON" \
  "$SCRIPT_DIR/annotate_normals_normalcrafter.py" \
  --normalcrafter-root "$NORMALCRAFTER_ROOT" \
  "$@"
