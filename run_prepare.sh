#!/usr/bin/env bash
# CPU: environment, scan/exclusion, transfer, unpack, align, overlap, shard plan.
set -Eeuo pipefail
exec bash "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/recam_refine/launchers/run.sh" prepare "$@"
