"""Read-only sidecar scan; Python standard library only. No image decoding."""
import argparse
import collections
import json
import math
import os
import sys
from pathlib import Path
try:
    from .progress import phase, run
except ImportError:
    from progress import phase, run


def inspect(path):
    result = {"path": str(path), "issues": [], "bad_pairs": []}
    try:
        before = path.stat()
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        s = data["source"]
        result.update({k: s.get(k) for k in (
            "episode_index", "source_episode_id", "camera_role", "camera_serial",
            "initial_decoded_frame_count", "tail_retry_attempted",
            "tail_retry_recovered_frames", "tail_missing_count")})
        ts = s["timestamps_ms"]
        if not isinstance(ts, list):
            raise ValueError("timestamps_ms must be a list")
        def integer(key):
            value = s[key]
            if type(value) is not int or value < 0:
                raise ValueError(f"Invalid {key}: {value!r}")
            return value
        n, decoded, missing = [integer(k) for k in (
            "frame_count", "decoded_frame_count", "tail_missing_count")]
        issues = result["issues"]
        if decoded <= 0 or decoded + missing != n or missing > 2:
            issues.append("invalid_frame_counts")
        if len(ts) != n:
            issues.append("timestamp_count_mismatch")
        if s.get("missing_frame_indices") != list(range(decoded, n)):
            issues.append("invalid_missing_indices")
        valid = lambda v: type(v) in (int, float) and math.isfinite(v)
        if any(not valid(v) for v in ts[:decoded]):
            issues.append("invalid_decoded_timestamp")
        if any(v is not None for v in ts[decoded:]):
            issues.append("invalid_padding_timestamp")
        for i in range(1, min(decoded, len(ts))):
            a, b = ts[i-1:i+1]
            if valid(a) and valid(b) and b <= a:
                kind = "equal" if a == b else "backward"
                result["bad_pairs"].append({"index": i, "previous_index": i-1,
                    "previous_ms": a, "current_ms": b, "kind": kind})
        if result["bad_pairs"]:
            issues.append("unordered_timestamps")
        initial = s.get("initial_decoded_frame_count")
        recovered = s.get("tail_retry_recovered_frames")
        if initial is not None or recovered is not None:
            if (type(initial) is not int or type(recovered) is not int
                    or initial < 0 or recovered < 0 or initial + recovered != decoded
                    or (recovered > 0 and s.get("tail_retry_attempted") is not True)):
                issues.append("inconsistent_retry_record")
        result["duplicate_final_retry_signature"] = bool(
            len(result["bad_pairs"]) == 1
            and result["bad_pairs"][0]["kind"] == "equal"
            and result["bad_pairs"][0]["index"] == decoded - 1
            and s.get("tail_retry_attempted") is True
            and recovered == 1 and missing == 0)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            issues.append("changed_during_read")
    except Exception as exc:
        result["issues"].append("read_or_schema_error")
        result["error"] = str(exc)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth-output", required=True, type=Path,
                        help="Conversion output root, or foundation_stereo_depth directory")
    parser.add_argument("--report", required=True, type=Path,
                        help="New JSON report outside the conversion output directory")
    parser.add_argument("--chunks", help="Optional chunk selection, e.g. 0-1 or 0,1,14-18")
    args = parser.parse_args()
    source = args.depth_output.resolve()
    root = source if source.name == "foundation_stereo_depth" else source / "annotations/foundation_stereo_depth"
    report = args.report.resolve()
    if source == report or source in report.parents:
        parser.error("Report must be outside --depth-output")
    if report.exists():
        parser.error("Report exists; choose a new report name")
    if os.environ.get('RECAM_PROGRESS_CHILD') != '1':
        return run([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]],
                   'timestamp-scan',report.with_suffix('.log'),capture=True)
    phase('扫描：定位 JSON',detail=str(root))
    files = sorted(root.glob("chunk-*/observation.images.depth_*/episode_*.json"))
    if args.chunks:
        try:
            selected = set()
            for part in args.chunks.split(","):
                bounds = [int(v) for v in part.split("-")]
                if len(bounds) == 1:
                    bounds *= 2
                if len(bounds) != 2 or not 0 <= bounds[0] <= bounds[1]:
                    raise ValueError()
                selected.update(f"chunk-{v:03d}" for v in range(bounds[0], bounds[1]+1))
        except ValueError:
            parser.error("Invalid --chunks; use 0-1 or 0,1,14-18")
        files = [p for p in files if p.relative_to(root).parts[0] in selected]
        absent = selected - {p.relative_to(root).parts[0] for p in files}
        if absent:
            parser.error(f"No JSON records for requested chunks: {sorted(absent)}")
    if not files:
        parser.error(f"No sidecars found under {root}")
    counts = collections.Counter({key: 0 for key in (
        "scanned_camera_records", "anomalous_camera_records",
        "duplicate_final_retry_records", "unordered_timestamps",
        "read_or_schema_error", "records_without_detected_anomalies")})
    episodes, affected, signature_episodes = set(), set(), set()
    anomalies = []
    by_chunk = {}
    phase('扫描：检查时间戳',0,len(files))
    for number,path in enumerate(files,1):
        r = inspect(path)
        chunk = path.relative_to(root).parts[0]
        bucket = by_chunk.setdefault(chunk, {"camera_records": 0, "episodes": set(),
            "affected_episodes": set(), "duplicate_final_retry_episodes": set()})
        bucket["camera_records"] += 1
        counts["scanned_camera_records"] += 1
        # Filename fallback keeps unreadable JSON visible in the episode list.
        episode = path.stem.removeprefix("episode_")
        episodes.add(episode)
        bucket["episodes"].add(episode)
        counts["records_with_retry"] += bool(r.get("tail_retry_attempted"))
        counts["records_with_padding"] += bool(r.get("tail_missing_count"))
        if r.get("duplicate_final_retry_signature"):
            counts["duplicate_final_retry_records"] += 1
            signature_episodes.add(episode)
            bucket["duplicate_final_retry_episodes"].add(episode)
        if r["issues"]:
            counts["anomalous_camera_records"] += 1
            affected.add(episode)
            bucket["affected_episodes"].add(episode)
            anomalies.append(r)
            for issue in r["issues"]:
                counts[issue] += 1
        else:
            counts["records_without_detected_anomalies"] += 1
        if number % 1000 == 0 or number == len(files):
            phase('扫描：检查时间戳',number,len(files),f'affected episodes={len(affected)}')
    summary = dict(counts)
    summary.update(scanned_episodes=len(episodes), affected_episodes=len(affected),
                   duplicate_final_retry_episodes=len(signature_episodes))
    chunk_summary = {c: {k: len(v) if isinstance(v, set) else v for k, v in b.items()}
                     for c, b in by_chunk.items()}
    payload = {"root": str(root), "summary": summary, "by_chunk": chunk_summary,
        "affected_episode_ids": sorted(affected),
        "duplicate_final_retry_episode_ids": sorted(signature_episodes),
        "limitations": "Sidecar-only scan. Cannot prove PNG alignment or detect every monotonic frame shift. Retry alone is not an error. No data deleted.",
        "anomalies": anomalies}
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(chunk_summary, ensure_ascii=False, indent=2))
    print(f"Report: {report}")


if __name__ == "__main__":
    sys.exit(main())
