#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
WORK_DIR="${NORMALCRAFTER_WORK_DIR:-$PROJECT_ROOT/Res/${NORMALCRAFTER_EXP_NAME:-default}}"
NORMALCRAFTER_ROOT="${NORMALCRAFTER_ROOT:-$WORK_DIR/NormalCrafter}"
ENV_NAME="${DROID_NORMALS_ENV_NAME:-droid_normals}"

if [[ -n "${MINIFORGE_HOME:-}" ]]; then
  export PATH="$MINIFORGE_HOME/bin:$PATH"
fi
if [[ -z "${NORMALCRAFTER_PYTHON:-}" ]] && command -v conda >/dev/null 2>&1; then
  NORMALCRAFTER_PYTHON="$(conda run -n "$ENV_NAME" python -c 'import sys; print(sys.executable)' 2>/dev/null || true)"
fi
NORMALCRAFTER_PYTHON="${NORMALCRAFTER_PYTHON:-}"
export HF_HOME="${HF_HOME:-$WORK_DIR/hf_cache}"

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
