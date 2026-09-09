#!/usr/bin/env bash
# GPU machine 1: fixed shard 0. Start only after run_prepare.sh succeeds.
set -Eeuo pipefail
exec bash "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/recam_refine/launchers/run.sh" gpu1 "$@"
