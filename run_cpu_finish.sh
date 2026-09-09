#!/usr/bin/env bash
# CPU: merge, apply, check, cleanup, repack. Both GPU shards must be complete.
set -Eeuo pipefail
exec bash "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/recam_refine/launchers/run.sh" finish "$@"
