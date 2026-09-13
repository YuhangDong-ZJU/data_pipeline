# Processing and resume behavior

The four machine scripts and their arguments are unchanged. Use the same work
directory when updating: existing completion records and GPU checkpoints remain
valid. Stop processes using the shared checkout before updating it.

The processing scripts no longer acquire filesystem locks, including environment
installation, exclusion, GPU workers, CPU validation and finalization. Old lock
files are ignored. Assign non-overlapping chunks/shards, start each shard once,
install shared dependencies on one CPU before workers start, and run CPU merging,
validation and finalization only after every GPU worker exits. Completion records
and recovery journals remain. Dependency version files named requirements*.lock
are package lists, not process/file locks.

| Stage | Work performed |
| --- | --- |
| Scan/exclude | Read timestamp sidecars and remove the confirmed episode with corresponding index/metadata updates. Completed exclusion is skipped. |
| Transfer | Move/copy selected camera streams concurrently. Count PNGs from the task results; do not reread all transfer receipts for a summary. |
| Unpack | Enumerate only requested chunks and extract TARs concurrently by camera directory. No transfer-receipt reads, PNG decoding, content hashing, disk-space estimate, or unpack locks. |
| Align | Read padding evidence, trim synchronized modalities, and update metadata/statistics. Preserve recovery records. |
| Shard plan | Assign episodes and prepare robot assets. New plans do not read depth PNGs or Parquet solely to hash them. |
| GPU workers | Read inputs needed for fitting, optimize and evaluate cameras, save results/checkpoints. Legacy input-hash fields are ignored. Checkpoints compare camera/frame/optimizer settings directly, without hashing PNGs, Parquet or robot points. Older checkpoint states are retained. |
| Merge/apply | Confirm shard coverage, combine accepted results, and update camera parameters. |
| Check | Decode media and check frame counts, metadata, statistics and geometry. No separate all-training-file fingerprint passes before/after this check. |
| Cleanup/repack | Require successful preceding stages; no additional all-training-file fingerprint scan. Retain simulation TARs and training PNGs when repacking DROID depth. |

Unpack assumes the partner layout: migrated DROID chunks 2–13 have PNGs and no
old depth TARs. It no longer consults migration records to resolve overlapping
old/new versions. A conflicting existing file causes an error rather than being
silently overwritten. Completed TARs still use their existing extraction records
and PNG existence/size to skip payload reads. PNG permissions and timestamps do not invalidate completion. Missing or wrong-size extracted files are restored from the same TAR; unchanged members are skipped. An interrupted TAR may be reread.

Do not modify training data between final check and cleanup/repack: those stages
now trust the successful check, rather than scanning all files again to detect
out-of-band changes. Checks that protect episode alignment, wrong shard results,
archive path traversal and incomplete writes remain. Existing source-copy cleanup
still compares data before deleting source files; this is a destructive action,
not an extra preflight. Shard code compatibility uses algorithm/schema versions rather than whole source-file hashes. Package download integrity checks and small plan/result
identity hashes also remain.

Bad depth images during calibration: retry decoding the same file once. If it
still cannot decode, record the filename and exclude that episode; other I/O,
CUDA and programming failures still fail the step. Fit rejection retains the
initial camera pose and does not exclude an episode. The fixed shard plan and
completed results remain reusable, including old results. Run GPU1 for shard 0
and GPU2 for shard 1, once per shard at a time.

The existing CPU `apply` step collects exclusions after both shards complete.
It moves excluded training episode entries to
`$RECAM_WORK/bad_depth_exclusion/episodes`, fills holes from surviving tail
episodes, and updates indices during the normal Parquet write pass. Metadata,
statistics, source identities and audit inputs follow the final mapping. GPU
plans/results remain unchanged. No extra PNG scan, content hash, or file lock
is added. Interrupted moves and apply results resume from their receipts.
Original converted depth copies for excluded/renumbered episodes are retained
outside the dataset. Obsolete DROID TARs are removed during the existing cleanup
step and the final TARs are generated from the retained episodes as usual.

After pulling this version, rerun the failed `run_gpu1.sh`/`run_gpu2.sh` launchers;
completed episodes are skipped. When both finish, run `run_cpu_finish.sh`.
Preparation does not need to be repeated just for this update.
