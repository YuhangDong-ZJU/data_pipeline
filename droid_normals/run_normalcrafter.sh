#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
NORMALCRAFTER_ROOT="${NORMALCRAFTER_ROOT:-/data2/normalcrafter_test/NormalCrafter}"
NORMALCRAFTER_PYTHON="${NORMALCRAFTER_PYTHON:-/data2/normalcrafter_test/env/bin/python}"
DEFAULT_NORMALCRAFTER_HF_HOME="/data2/normalcrafter_test/hf_cache"

if [[ -z "${HF_HOME:-}" && -d "$DEFAULT_NORMALCRAFTER_HF_HOME" ]]; then
  export HF_HOME="$DEFAULT_NORMALCRAFTER_HF_HOME"
fi

if [[ ! -x "$NORMALCRAFTER_PYTHON" ]]; then
  echo "ERROR: NormalCrafter Python was not found: $NORMALCRAFTER_PYTHON" >&2
  echo "Set NORMALCRAFTER_PYTHON to the Python executable in its environment." >&2
  exit 1
fi

if [[ ! -f "$NORMALCRAFTER_ROOT/normalcrafter/normal_crafter_ppl.py" ]]; then
  echo "ERROR: NormalCrafter checkout was not found: $NORMALCRAFTER_ROOT" >&2
  echo "Set NORMALCRAFTER_ROOT or follow $ROOT/README.md." >&2
  exit 1
fi

exec "$NORMALCRAFTER_PYTHON" \
  "$ROOT/annotate_normals_normalcrafter.py" \
  --normalcrafter-root "$NORMALCRAFTER_ROOT" \
  "$@"
