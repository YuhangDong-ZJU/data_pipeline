#!/usr/bin/env bash
set -Eeuo pipefail
export PYTHONNOUSERSITE=1
unset PYTHONHOME PYTHONPATH
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
if [[ $# -lt 3 ]]; then
  echo 'Usage: bash recam_refine/run_step.sh <transfer|unpack|align|overlap|refine|shard-plan|shard-refine|shard-merge|shard-status|apply|check|cleanup|repack|status> <recam_lerobot> <work_dir> [--reuse-env | --python PATH | --conda-env NAME] [--prepared-runtime] [--runtime-work-dir SHARED_ENV (shard-refine only)] [options]' >&2
  exit 2
fi
STEP="$1"
DATASET="$2"
WORK_DIR="$3"
shift 3
PREPARED=()
SHARED_RUNTIME=""
REUSE_ENV="${RECAM_REFINE_REUSE_ENV:-0}"
REUSE_SELECTOR=()
if [[ -n "${RECAM_REFINE_PYTHON:-}" || -n "${RECAM_REFINE_ENV_NAME:-}" ]]; then REUSE_ENV=1; fi
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --reuse-env) REUSE_ENV=1; shift ;;
    --python|--conda-env)
      if [[ $# -lt 2 || -z "$2" || "$2" == --* || ${#REUSE_SELECTOR[@]} -gt 0 ]]; then
        echo 'ERROR: supply exactly one --python PATH or --conda-env NAME.' >&2; exit 2
      fi
      REUSE_ENV=1; REUSE_SELECTOR=("$1" "$2"); shift 2 ;;
    --python=*|--conda-env=*)
      if [[ -z "${1#*=}" || ${#REUSE_SELECTOR[@]} -gt 0 ]]; then
        echo 'ERROR: supply exactly one --python PATH or --conda-env NAME.' >&2; exit 2
      fi
      REUSE_ENV=1; REUSE_SELECTOR=("$1"); shift ;;
    --prepared-runtime) PREPARED=(--verify-only); shift ;;
    --runtime-work-dir)
      if [[ $# -lt 2 || -z "$2" || "$2" == --* || -n "$SHARED_RUNTIME" ]]; then
        echo 'ERROR: --runtime-work-dir requires one nonempty path, supplied once.' >&2; exit 2
      fi
      SHARED_RUNTIME="$2"; shift 2 ;;
    --runtime-work-dir=*)
      if [[ -n "$SHARED_RUNTIME" || -z "${1#*=}" ]]; then
        echo 'ERROR: --runtime-work-dir requires one nonempty path, supplied once.' >&2; exit 2
      fi
      SHARED_RUNTIME="${1#*=}"; shift ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done
set -- "${EXTRA[@]}"
if [[ "$REUSE_ENV" != 0 && "$REUSE_ENV" != 1 ]]; then
  echo 'ERROR: RECAM_REFINE_REUSE_ENV must be 0 or 1.' >&2; exit 2
fi
if [[ "$REUSE_ENV" == 1 && -n "$SHARED_RUNTIME" ]]; then
  echo 'ERROR: --reuse-env selects an existing environment; omit --runtime-work-dir, or use --python with its env/bin/python.' >&2; exit 2
fi
if [[ "$REUSE_ENV" == 0 ]]; then unset LD_PRELOAD LD_LIBRARY_PATH; fi
case "$STEP" in exclude-6795|transfer|unpack|align|overlap|refine|shard-plan|shard-refine|shard-merge|shard-status|apply|check|cleanup|repack|status) ;; *) echo "Unknown step: $STEP" >&2; exit 2 ;; esac
if [[ -n "$SHARED_RUNTIME" && "$STEP" != shard-refine ]]; then
  echo 'ERROR: --runtime-work-dir is supported only by shard-refine; keep the CPU coordinator runtime separate.' >&2; exit 2
fi
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
fi
if [[ "$STEP" == shard-refine || "$STEP" == refine ]]; then
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
LOG_WORK="$RUNTIME_WORK"
if [[ -n "$SHARED_RUNTIME" ]]; then
  RUNTIME_WORK="$(python3 - "$SHARED_RUNTIME" "$DATASET" "$WORK_DIR" "$LOG_WORK" <<'PY'
from pathlib import Path
import sys
runtime = Path(sys.argv[1]).expanduser().resolve()
actual = (runtime/'runtime').resolve()
for value in sys.argv[2:]:
    other = Path(value).expanduser().resolve()
    if any(a.is_relative_to(other) or other.is_relative_to(a) for a in (runtime,actual)):
        raise SystemExit('ERROR: shared runtime must be separate from dataset, coordinator and worker directories, including symlink targets.')
if not (actual/'bootstrap.lock').is_file():
    raise SystemExit('ERROR: shared runtime not prepared; run bootstrap.py on the CPU host first.')
print(runtime)
PY
)"
fi
mkdir -p "$WORK_DIR"
WORK_DIR="$(cd "$WORK_DIR" && pwd)"
if [[ "$STEP" == exclude-6795 ]]; then
  if [[ "$REUSE_ENV" == 1 ]]; then
    python3 -m recam_refine.environment "$WORK_DIR" --profile base "${REUSE_SELECTOR[@]}" --exec \
      -m recam_refine.exclude_episode --root "$DATASET" --work-dir "$WORK_DIR" "$@" \
      2>&1 | tee -a "$WORK_DIR/step_exclude_6795.log"
  else
    python3 recam_refine/bootstrap.py "$WORK_DIR" "${PREPARED[@]}" --exec \
      -m recam_refine.exclude_episode --root "$DATASET" --work-dir "$WORK_DIR" "$@" \
      2>&1 | tee -a "$WORK_DIR/step_exclude_6795.log"
  fi
  exit 0
fi
if [[ "$REUSE_ENV" == 1 ]]; then
  PROFILE=base
  case "$STEP" in
    overlap) PROFILE=overlap ;;
    shard-plan|shard-merge) PROFILE=geometry ;;
    check) PROFILE=cpu ;;
    refine|shard-refine)
      if [[ "$DEVICES" == cpu ]]; then PROFILE=cpu; else PROFILE=gpu; fi ;;
  esac
  mkdir -p "$LOG_WORK"
  COMMAND=(run-step "$STEP")
  if [[ "$STEP" == transfer ]]; then COMMAND=(transfer-depth); fi
  python3 -m recam_refine.environment "$WORK_DIR" --profile "$PROFILE" \
    --cache-work-dir "$LOG_WORK" "${REUSE_SELECTOR[@]}" --exec \
    -m recam_refine "${COMMAND[@]}" "$DATASET" --work-dir "$WORK_DIR" "$@" \
    2>&1 | tee -a "$LOG_WORK/step_${STEP}.log"
  exit 0
fi
BOOTSTRAP=()
if [[ "$STEP" == check ]]; then
  BOOTSTRAP+=(--cpu-torch)
elif [[ "$STEP" == refine || "$STEP" == shard-refine ]]; then
  if [[ "$DEVICES" == cpu ]]; then BOOTSTRAP+=(--cpu-torch); else BOOTSTRAP+=(--gpu); fi
fi
if [[ -n "$SHARED_RUNTIME" ]]; then
  mkdir -p "$LOG_WORK"
  # Keep installation/bytecode shared files unchanged; each worker owns its
  # mutable driver/compiler caches as well as its logs and calibration state.
  export PYTHONDONTWRITEBYTECODE=1
  export XDG_CACHE_HOME="$LOG_WORK/runtime_cache/xdg"
  export CUDA_CACHE_PATH="$LOG_WORK/runtime_cache/cuda"
  export TRITON_CACHE_DIR="$LOG_WORK/runtime_cache/triton"
  export TORCHINDUCTOR_CACHE_DIR="$LOG_WORK/runtime_cache/torchinductor"
  export TORCH_EXTENSIONS_DIR="$LOG_WORK/runtime_cache/torch_extensions"
  mkdir -p "$XDG_CACHE_HOME" "$CUDA_CACHE_PATH" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"
  echo "Shared prepared runtime: $RUNTIME_WORK/runtime; worker: $LOG_WORK"
  python3 recam_refine/bootstrap.py "$RUNTIME_WORK" "${BOOTSTRAP[@]}" --verify-only --exec \
    -m recam_refine run-step "$STEP" "$DATASET" --work-dir "$WORK_DIR" "$@" \
    2>&1 | tee -a "$LOG_WORK/step_${STEP}.log"
  exit 0
fi
mkdir -p "$RUNTIME_WORK"
python3 recam_refine/bootstrap.py "$RUNTIME_WORK" "${BOOTSTRAP[@]}" "${PREPARED[@]}" 2>&1 | tee -a "$RUNTIME_WORK/install.log"
if [[ "$STEP" == transfer ]]; then
  "$WORK_DIR/runtime/env/bin/python" -m recam_refine transfer-depth "$DATASET" --work-dir "$WORK_DIR" "$@" \
    2>&1 | tee -a "$WORK_DIR/step_transfer.log"
else
  "$RUNTIME_WORK/runtime/env/bin/python" -m recam_refine run-step "$STEP" "$DATASET" --work-dir "$WORK_DIR" "$@" \
    2>&1 | tee -a "$RUNTIME_WORK/step_${STEP}.log"
fi
