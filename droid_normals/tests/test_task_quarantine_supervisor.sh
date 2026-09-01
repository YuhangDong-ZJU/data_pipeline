#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
NORMALS_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
TEST_ROOT="$(mktemp -d)"
trap 'rm -rf -- "$TEST_ROOT"' EXIT

mkdir -p \
  "$TEST_ROOT/bin" \
  "$TEST_ROOT/dataset" \
  "$TEST_ROOT/runtime/NormalCrafter/normalcrafter" \
  "$TEST_ROOT/run"
touch "$TEST_ROOT/runtime/NormalCrafter/normalcrafter/normal_crafter_ppl.py"

cat > "$TEST_ROOT/bin/nvidia-smi" <<'SH'
#!/usr/bin/env bash
exit 0
SH
chmod +x "$TEST_ROOT/bin/nvidia-smi"

cat > "$TEST_ROOT/fake_worker.py" <<'PY'
import argparse
import os
from pathlib import Path

from normal_task_recovery import write_active_task

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--active-task-file", type=Path, required=True)
args, _ = parser.parse_known_args()

state_dir = Path(os.environ["DROID_NORMALS_TEST_STATE"])
counter = state_dir / "launches.count"
launch = int(counter.read_text()) + 1 if counter.exists() else 1
counter.write_text(f"{launch}\n")

if launch <= 3:
    write_active_task(
        args.active_task_file,
        {
            "subset": "real_world/droid",
            "chunk": "chunk-003",
            "input_camera": "observation.images.rgb_01",
            "output_camera": "observation.images.normal_01",
            "episode": "episode_003122",
            "input_path": "/dataset/rgb/episode_003122.mp4",
            "output_path": "/dataset/normal/episode_003122.mp4",
            "metadata_path": "/logs/episode_003122.json",
        },
    )
    os._exit(139)
PY

export PATH="$TEST_ROOT/bin:$PATH"
export PYTHONPATH="$NORMALS_DIR"
export DROID_NORMALS_TEST_STATE="$TEST_ROOT"
export DROID_NORMALS_SKIP_INSTALL=1
export NORMALCRAFTER_PYTHON="$(command -v python3)"
export DROID_NORMALS_WORKER="$TEST_ROOT/fake_worker.py"
export DROID_NORMALS_ATTENTION_BACKEND=pytorch
export DROID_NORMALS_DRY_RUN=1
export DROID_NORMALS_WORKER_PASSES=1
export DROID_NORMALS_NATIVE_CRASH_MAX_ATTEMPTS=3

if bash "$NORMALS_DIR/run_droid_normals_conversion.sh" \
  3 \
  "$TEST_ROOT/dataset" \
  0 \
  "$TEST_ROOT/runtime" \
  "$TEST_ROOT/run" \
  01 \
  1; then
  echo "Expected a final nonzero status for the quarantined video." >&2
  exit 1
fi

[[ "$(<"$TEST_ROOT/launches.count")" == "4" ]]
record="$(find "$TEST_ROOT/run/recovery" -path '*/tasks/*.json' -type f -print -quit)"
[[ -n "$record" ]]
grep -q '"status": "quarantined"' "$record"
grep -q '"native_crash_attempts": 3' "$record"

echo "Tracked native-crash quarantine test passed."
