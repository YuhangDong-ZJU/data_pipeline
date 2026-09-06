#!/usr/bin/env bash
set -Eeuo pipefail
export PYTHONNOUSERSITE=1
unset PYTHONHOME PYTHONPATH LD_PRELOAD LD_LIBRARY_PATH
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
if [[ $# -lt 3 ]]; then
  echo 'Usage: bash recam_refine/run_step.sh <transfer|unpack|align|overlap|refine|apply|check|cleanup|status> <recam_lerobot> <work_dir> [options]' >&2
  exit 2
fi
STEP="$1"
DATASET="$2"
WORK_DIR="$3"
shift 3
case "$STEP" in transfer|unpack|align|overlap|refine|apply|check|cleanup|status) ;; *) echo "Unknown step: $STEP" >&2; exit 2 ;; esac
python3 - "$DATASET" "$WORK_DIR" "$STEP" "$@" <<'PY'
import argparse
from pathlib import Path
import sys
root,work = (Path(p).expanduser().resolve() for p in sys.argv[1:3])
if not (root/'real_world/droid').is_dir() or work.is_relative_to(root) or root.is_relative_to(work):
    raise SystemExit('ERROR: dataset/real_world/droid must exist; work_dir must be separate and outside the dataset.')
if sys.argv[3]=='transfer':
    parser = argparse.ArgumentParser()
    parser.add_argument('--depth-output',type=Path,required=True)
    args,_ = parser.parse_known_args(sys.argv[4:])
    source=args.depth_output.expanduser().resolve()
    if not source.is_dir() or any(a.is_relative_to(b) or b.is_relative_to(a) for a,b in ((source,root),(source,work))):
        raise SystemExit('ERROR: depth_output must exist and be separate from dataset and work_dir.')
PY
mkdir -p "$WORK_DIR"
WORK_DIR="$(cd "$WORK_DIR" && pwd)"
BOOTSTRAP=()
if [[ "$STEP" == refine || "$STEP" == check ]]; then BOOTSTRAP+=(--gpu); fi
python3 recam_refine/bootstrap.py "$WORK_DIR" "${BOOTSTRAP[@]}" 2>&1 | tee -a "$WORK_DIR/install.log"
if [[ "$STEP" == transfer ]]; then
  "$WORK_DIR/runtime/env/bin/python" -m recam_refine transfer-depth "$DATASET" --work-dir "$WORK_DIR" "$@" \
    2>&1 | tee -a "$WORK_DIR/step_transfer.log"
else
  "$WORK_DIR/runtime/env/bin/python" -m recam_refine run-step "$STEP" "$DATASET" --work-dir "$WORK_DIR" "$@" \
    2>&1 | tee -a "$WORK_DIR/step_${STEP}.log"
fi
