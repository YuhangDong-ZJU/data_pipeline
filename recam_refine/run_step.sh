#!/usr/bin/env bash
set -Eeuo pipefail
export PYTHONNOUSERSITE=1
unset PYTHONHOME PYTHONPATH LD_PRELOAD LD_LIBRARY_PATH
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
if [[ $# -lt 3 ]]; then
  echo 'Usage: bash recam_refine/run_step.sh <transfer|unpack|align|overlap|refine|shard-plan|shard-refine|shard-merge|shard-status|apply|check|cleanup|status> <recam_lerobot> <work_dir> [options]' >&2
  exit 2
fi
STEP="$1"
DATASET="$2"
WORK_DIR="$3"
shift 3
case "$STEP" in transfer|unpack|align|overlap|refine|shard-plan|shard-refine|shard-merge|shard-status|apply|check|cleanup|status) ;; *) echo "Unknown step: $STEP" >&2; exit 2 ;; esac
RUNTIME_WORK="$WORK_DIR"
DEVICES="0,1,2,3,4,5,6,7"
if [[ "$STEP" == shard-refine ]]; then
  # Parse with the system Python before installing anything. Workers must not
  # concurrently install dependencies into the shared coordinator runtime.
  RUNTIME_WORK="$(python3 - "$DATASET" "$WORK_DIR" "$@" <<'PY'
import argparse
from pathlib import Path
import sys
p=argparse.ArgumentParser()
p.add_argument('--worker-work-dir',type=Path,required=True)
p.add_argument('--shard-id',type=int,required=True)
a,_=p.parse_known_args(sys.argv[3:])
root,work=(Path(x).expanduser().resolve() for x in sys.argv[1:3])
local=a.worker_work_dir.expanduser().resolve()
if not (work/'SHARD_PLAN_READY.json').is_file() or a.shard_id<0:
    raise SystemExit('ERROR: run shard-plan first and supply a valid zero-based --shard-id.')
if any(a.is_relative_to(b) or b.is_relative_to(a) for a,b in ((root,local),(work,local))):
    raise SystemExit('ERROR: worker-work-dir must be separate from dataset and coordinator, without nesting.')
print(local)
PY
)"
  DEVICES="$(python3 - "$@" <<'PY'
import argparse
p=argparse.ArgumentParser()
p.add_argument('--devices',default='0,1,2,3,4,5,6,7')
a,_=p.parse_known_args()
print(a.devices)
PY
)"
fi
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
if [[ "$STEP" == refine || "$STEP" == check || ( "$STEP" == shard-refine && "$DEVICES" != cpu ) ]]; then BOOTSTRAP+=(--gpu); fi
mkdir -p "$RUNTIME_WORK"
python3 recam_refine/bootstrap.py "$RUNTIME_WORK" "${BOOTSTRAP[@]}" 2>&1 | tee -a "$RUNTIME_WORK/install.log"
if [[ "$STEP" == transfer ]]; then
  "$WORK_DIR/runtime/env/bin/python" -m recam_refine transfer-depth "$DATASET" --work-dir "$WORK_DIR" "$@" \
    2>&1 | tee -a "$WORK_DIR/step_transfer.log"
else
  "$RUNTIME_WORK/runtime/env/bin/python" -m recam_refine run-step "$STEP" "$DATASET" --work-dir "$WORK_DIR" "$@" \
    2>&1 | tee -a "$RUNTIME_WORK/step_${STEP}.log"
fi
