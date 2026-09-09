#!/usr/bin/env bash
# GPU machine 2: fixed shard 1. May run concurrently with run_gpu1.sh.
set -Eeuo pipefail
exec bash "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/recam_refine/launchers/run.sh" gpu2 "$@"
