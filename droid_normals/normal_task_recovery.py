#!/usr/bin/env python3
"""Persistent recovery state for NormalCrafter worker crashes.

The GPU worker writes one small active-task marker before it starts decoding a
video.  If the process dies in native code, the shell supervisor records that
marker here, retries the same task, and quarantines only that task after a
bounded number of process crashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any


STATE_VERSION = 1


def task_key(identity: dict[str, Any]) -> str:
    stable = {
        "subset": str(identity["subset"]),
        "chunk": str(identity["chunk"]),
        "input_camera": str(identity["input_camera"]),
        "episode": str(identity["episode"]),
    }
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.part")
    with part.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(part, path)


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def record_path(state_dir: Path, key: str) -> Path:
    return state_dir / "tasks" / f"{key}.json"


def append_event(state_dir: Path, payload: dict[str, Any]) -> None:
    path = state_dir / "events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(descriptor, encoded)
    finally:
        os.close(descriptor)


def write_active_task(path: Path, identity: dict[str, Any]) -> str:
    key = task_key(identity)
    payload = {
        "version": STATE_VERSION,
        "task_key": key,
        "task": identity,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "started_unix": time.time(),
    }
    write_json_atomic(path, payload)
    return key


def clear_active_task(
    path: Path,
    *,
    expected_key: str | None = None,
    expected_pid: int | None = None,
) -> None:
    payload = read_json(path)
    if payload is None:
        path.unlink(missing_ok=True)
        return
    if expected_key is not None and payload.get("task_key") != expected_key:
        return
    if expected_pid is not None and int(payload.get("pid", -1)) != expected_pid:
        return
    path.unlink(missing_ok=True)


def load_quarantined_keys(state_dir: Path | None) -> set[str]:
    if state_dir is None:
        return set()
    keys: set[str] = set()
    for path in (state_dir / "tasks").glob("*.json"):
        payload = read_json(path)
        if payload is not None and payload.get("status") == "quarantined":
            keys.add(str(payload.get("task_key", path.stem)))
    return keys


def mark_resolved(state_dir: Path, key: str) -> None:
    path = record_path(state_dir, key)
    payload = read_json(path)
    if payload is None or payload.get("status") == "resolved":
        return
    payload.update({"status": "resolved", "resolved_unix": time.time()})
    write_json_atomic(path, payload)
    append_event(
        state_dir,
        {"event": "resolved", "task_key": key, "time": time.time()},
    )


def quarantine_python_failure(
    state_dir: Path,
    identity: dict[str, Any],
    *,
    attempts: int,
    error_type: str,
    message: str,
) -> dict[str, Any]:
    key = task_key(identity)
    path = record_path(state_dir, key)
    previous = read_json(path) or {}
    payload = {
        "version": STATE_VERSION,
        "status": "quarantined",
        "task_key": key,
        "task": identity,
        "native_crash_attempts": int(previous.get("native_crash_attempts", 0)),
        "python_attempts": attempts,
        "last_failure_kind": "python_exception",
        "last_error_type": error_type,
        "last_message": message,
        "updated_unix": time.time(),
    }
    write_json_atomic(path, payload)
    append_event(
        state_dir,
        {
            "event": "quarantined",
            "failure_kind": "python_exception",
            "task_key": key,
            "attempts": attempts,
            "time": time.time(),
        },
    )
    return payload


def record_native_crash(
    state_dir: Path,
    active_file: Path,
    *,
    max_attempts: int,
    exit_status: int,
    gpu_id: str,
    shard_index: int,
    worker_pid: int,
) -> dict[str, Any] | None:
    active = read_json(active_file)
    if active is None:
        return None
    if int(active.get("pid", -1)) != worker_pid:
        return None
    identity = active.get("task")
    key = active.get("task_key")
    if not isinstance(identity, dict) or not isinstance(key, str) or key != task_key(identity):
        return None

    path = record_path(state_dir, key)
    previous = read_json(path) or {}
    previous_attempts = (
        int(previous.get("native_crash_attempts", 0))
        if previous.get("status") != "resolved"
        else 0
    )
    attempts = previous_attempts + 1
    quarantined = attempts >= max_attempts
    payload = {
        "version": STATE_VERSION,
        "status": "quarantined" if quarantined else "retrying",
        "task_key": key,
        "task": identity,
        "native_crash_attempts": attempts,
        "native_crash_max_attempts": max_attempts,
        "last_failure_kind": "process_crash",
        "last_exit_status": exit_status,
        "last_gpu_id": gpu_id,
        "last_shard_index": shard_index,
        "last_worker_pid": worker_pid,
        "updated_unix": time.time(),
    }
    write_json_atomic(path, payload)
    append_event(
        state_dir,
        {
            "event": "quarantined" if quarantined else "retrying",
            "failure_kind": "process_crash",
            "task_key": key,
            "attempt": attempts,
            "max_attempts": max_attempts,
            "exit_status": exit_status,
            "gpu_id": gpu_id,
            "shard_index": shard_index,
            "worker_pid": worker_pid,
            "time": time.time(),
        },
    )
    clear_active_task(active_file, expected_key=key, expected_pid=worker_pid)
    return payload


def unresolved_quarantines(state_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted((state_dir / "tasks").glob("*.json")):
        payload = read_json(path)
        if payload is not None and payload.get("status") == "quarantined":
            records.append(payload)
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage NormalCrafter crash recovery state")
    subparsers = parser.add_subparsers(dest="command", required=True)

    record = subparsers.add_parser("record-crash")
    record.add_argument("--state-dir", type=Path, required=True)
    record.add_argument("--active-file", type=Path, required=True)
    record.add_argument("--max-attempts", type=int, required=True)
    record.add_argument("--exit-status", type=int, required=True)
    record.add_argument("--gpu-id", required=True)
    record.add_argument("--shard-index", type=int, required=True)
    record.add_argument("--worker-pid", type=int, required=True)

    summary = subparsers.add_parser("summary")
    summary.add_argument("--state-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "record-crash":
        if args.max_attempts < 1:
            raise ValueError("--max-attempts must be positive")
        result = record_native_crash(
            args.state_dir,
            args.active_file,
            max_attempts=args.max_attempts,
            exit_status=args.exit_status,
            gpu_id=args.gpu_id,
            shard_index=args.shard_index,
            worker_pid=args.worker_pid,
        )
        if result is None:
            print("UNTRACKED_CRASH: no matching active task marker", flush=True)
            return 3
        task = result["task"]
        action = "quarantine" if result["status"] == "quarantined" else "retry"
        print(
            "TRACKED_CRASH: "
            f"{task['chunk']} | {task['input_camera']} | {task['episode']} | "
            f"attempt {result['native_crash_attempts']}/{result['native_crash_max_attempts']} | "
            f"action={action}",
            flush=True,
        )
        return 0

    records = unresolved_quarantines(args.state_dir)
    if not records:
        print("Recovery summary: no quarantined videos.", flush=True)
        return 0
    print(
        f"Recovery summary: {len(records)} video(s) quarantined; "
        f"records: {args.state_dir / 'tasks'}",
        file=sys.stderr,
        flush=True,
    )
    for payload in records[:50]:
        task = payload["task"]
        print(
            f"QUARANTINED: {task['chunk']} | {task['input_camera']} | "
            f"{task['episode']} | {payload.get('last_failure_kind')}",
            file=sys.stderr,
            flush=True,
        )
    if len(records) > 50:
        print(f"... and {len(records) - 50} more", file=sys.stderr, flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
