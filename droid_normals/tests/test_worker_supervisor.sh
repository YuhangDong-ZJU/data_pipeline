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

cat > "$TEST_ROOT/fake_worker.sh" <<'SH'
#!/usr/bin/env bash
set -Eeuo pipefail
state_dir="${DROID_NORMALS_TEST_STATE:?}"
gpu="${CUDA_VISIBLE_DEVICES:?}"
counter="$state_dir/gpu-$gpu.count"
attempt=1
if [[ -f "$counter" ]]; then
  attempt=$(( $(<"$counter") + 1 ))
fi
printf '%s\n' "$attempt" > "$counter"
printf 'gpu%s-attempt%s-start\n' "$gpu" "$attempt" >> "$state_dir/events"

if [[ "$gpu" == "0" && "$attempt" == "1" ]]; then
  sleep 0.2
  printf 'gpu0-attempt1-fail\n' >> "$state_dir/events"
  exit 139
fi
if [[ "$gpu" == "1" ]]; then
  sleep 2
fi
printf 'gpu%s-attempt%s-done\n' "$gpu" "$attempt" >> "$state_dir/events"
SH
chmod +x "$TEST_ROOT/fake_worker.sh"

export PATH="$TEST_ROOT/bin:$PATH"
export DROID_NORMALS_TEST_STATE="$TEST_ROOT"
export DROID_NORMALS_SKIP_INSTALL=1
export NORMALCRAFTER_PYTHON=/bin/bash
export DROID_NORMALS_WORKER="$TEST_ROOT/fake_worker.sh"
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
