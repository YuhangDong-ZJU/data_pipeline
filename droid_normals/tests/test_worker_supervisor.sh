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
import os
import sys
import time
from pathlib import Path

state_dir = Path(os.environ["DROID_NORMALS_TEST_STATE"])
gpu = os.environ["CUDA_VISIBLE_DEVICES"]
counter = state_dir / f"gpu-{gpu}.count"
attempt = int(counter.read_text()) + 1 if counter.exists() else 1
counter.write_text(f"{attempt}\n")
with (state_dir / "events").open("a") as events:
    events.write(f"gpu{gpu}-attempt{attempt}-start\n")

if gpu == "0" and attempt == 1:
    time.sleep(0.2)
    with (state_dir / "events").open("a") as events:
        events.write("gpu0-attempt1-fail\n")
    raise SystemExit(139)
if gpu == "1":
    time.sleep(2)
with (state_dir / "events").open("a") as events:
    events.write(f"gpu{gpu}-attempt{attempt}-done\n")
PY

export PATH="$TEST_ROOT/bin:$PATH"
export DROID_NORMALS_TEST_STATE="$TEST_ROOT"
export DROID_NORMALS_SKIP_INSTALL=1
export NORMALCRAFTER_PYTHON="$(command -v python3)"
export DROID_NORMALS_WORKER="$TEST_ROOT/fake_worker.py"
export DROID_NORMALS_ATTENTION_BACKEND=pytorch
export DROID_NORMALS_DRY_RUN=1
export DROID_NORMALS_WORKER_PASSES=2
export DROID_NORMALS_GLOBAL_NUM_WORKERS=2

bash "$NORMALS_DIR/run_droid_normals_conversion.sh" \
  0 \
  "$TEST_ROOT/dataset" \
  0,1 \
  "$TEST_ROOT/runtime" \
  "$TEST_ROOT/run" \
  01 \
  1

[[ "$(<"$TEST_ROOT/gpu-0.count")" == "2" ]]
[[ "$(<"$TEST_ROOT/gpu-1.count")" == "1" ]]
restart_line="$(grep -n '^gpu0-attempt2-start$' "$TEST_ROOT/events" | cut -d: -f1)"
other_done_line="$(grep -n '^gpu1-attempt1-done$' "$TEST_ROOT/events" | cut -d: -f1)"
if (( restart_line >= other_done_line )); then
  echo "GPU 0 was not restarted while GPU 1 was still active." >&2
  cat "$TEST_ROOT/events" >&2
  exit 1
fi

echo "Independent GPU worker restart test passed."
