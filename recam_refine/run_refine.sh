#!/usr/bin/env bash
set -Eeuo pipefail
export PYTHONNOUSERSITE=1
unset PYTHONHOME PYTHONPATH LD_PRELOAD LD_LIBRARY_PATH
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
if [[ $# -lt 2 ]]; then
  echo "Usage: bash recam_refine/run_refine.sh <recam_lerobot> <work_dir> [run options]" >&2
  exit 2
fi
DATASET="$1"
WORK_DIR="$2"
shift 2
python3 - "$DATASET" "$WORK_DIR" <<'PY'
from pathlib import Path
import sys
root, work = (Path(p).expanduser().resolve() for p in sys.argv[1:])
if not root.is_dir() or work.is_relative_to(root) or root.is_relative_to(work):
    raise SystemExit("ERROR: dataset must exist; work_dir must be separate and outside the dataset.")
PY
mkdir -p "$WORK_DIR"
WORK_DIR="$(cd "$WORK_DIR" && pwd)"
python3 recam_refine/bootstrap.py "$WORK_DIR" --gpu 2>&1 | tee -a "$WORK_DIR/install.log"
"$WORK_DIR/runtime/env/bin/python" -m recam_refine run "$DATASET" --work-dir "$WORK_DIR" "$@" --defer-cleanup 2>&1 | tee -a "$WORK_DIR/run.log"
# Every episode gets the paper's geometric metrics. Detailed image galleries
# can be generated for chosen episodes with audit-cameras (documented below).
"$WORK_DIR/runtime/env/bin/python" -m recam_refine audit-cameras "$DATASET" \
  --work-dir "$WORK_DIR" --report-dir "$WORK_DIR/camera_audit" \
  --episodes all --frames 8 --image-frames 0 --workers 4 --device cpu --fail-on-review \
  2>&1 | tee -a "$WORK_DIR/camera_audit.log"
# All checks passed; replay skips completed stages and only finalizes cleanup.
"$WORK_DIR/runtime/env/bin/python" -m recam_refine run "$DATASET" --work-dir "$WORK_DIR" "$@" 2>&1 | tee -a "$WORK_DIR/run.log"
