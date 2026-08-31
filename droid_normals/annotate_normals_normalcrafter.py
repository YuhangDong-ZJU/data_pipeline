#!/usr/bin/env python3
"""Annotate LeRobot RGB videos with NormalCrafter surface-normal MP4s."""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import os
import queue
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


_STOP = object()
_ABORT = object()
_PAGE_CACHE_WARNING_LOCK = threading.Lock()
_PAGE_CACHE_WARNINGS: set[str] = set()
NORMALCRAFTER_COMMIT = "75af9887a2cb14cd1ce3883c5773bc296565777c"
DEFAULT_UNET_PATH = "Yanrui95/NormalCrafter"
DEFAULT_UNET_REVISION = "7e24d68d86ae008fe08ef50b4e51cd2fc2c8cf57"
DEFAULT_PRETRAIN_PATH = "stabilityai/stable-video-diffusion-img2vid-xt"
DEFAULT_PRETRAIN_REVISION = "9e43909513c6714f1bc78bcb44d96e733cd242aa"
ANNOTATION_CONFIG_VERSION = 1


@dataclass(frozen=True)
class NormalTask:
    subset_root: Path
    subset_relative: Path
    chunk: str
    input_camera: str
    output_camera: str
    episode: str
    input_path: Path
    output_path: Path
    metadata_path: Path


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    frames: int
    fps: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate observation.images.normal_* MP4s from LeRobot RGB videos "
            "with one persistent NormalCrafter model."
        )
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Separate output root; default: write normal_* beside rgb_* in the dataset",
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        help="Metadata/log root; default: <output-root>/normalcrafter_logs",
    )
    parser.add_argument(
        "--normalcrafter-root",
        type=Path,
        required=True,
        help="Checkout of Binyr/NormalCrafter with normalcrafter_long_video.patch applied",
    )
    parser.add_argument("--subsets", nargs="+", help="Subset names or paths relative to dataset_root")
    parser.add_argument(
        "--chunks",
        nargs="+",
        help="Chunks such as 0,2,5-7 or chunk-000 chunk-002",
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        help="Camera selectors such as 01, rgb_01, or observation.images.rgb_01; default: all",
    )
    parser.add_argument(
        "--episodes",
        nargs="+",
        help="Episode selectors such as 12 or episode_000012",
    )
    parser.add_argument("--episode-start", type=int, help="Inclusive numeric episode lower bound")
    parser.add_argument("--episode-end", type=int, help="Inclusive numeric episode upper bound")
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total deterministic worker shards across all machines and GPUs",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based global shard assigned to this GPU worker",
    )
    parser.add_argument("--limit", type=int, help="Process at most this many selected videos")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-limit", type=int, default=50)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--verbose-inference",
        action="store_true",
        help="Show NormalCrafter internals and progress bars; default: concise worker output",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Maximum attempts for each video, including the first attempt",
    )
    parser.add_argument(
        "--retry-delay-seconds",
        type=float,
        default=5.0,
        help="Delay before retrying a failed video",
    )
    parser.add_argument(
        "--stale-lock-hours",
        type=float,
        default=0.25,
        help=(
            "Replace remote/unknown output locks whose heartbeat is older than this many "
            "hours; dead same-host owners are reclaimed immediately; 0 disables age fallback"
        ),
    )
    parser.add_argument(
        "--lock-heartbeat-seconds",
        type=float,
        default=30.0,
        help="Refresh owned output locks at this interval; 0 disables heartbeats",
    )

    parser.add_argument("--unet-path", default=DEFAULT_UNET_PATH)
    parser.add_argument("--unet-revision", default=DEFAULT_UNET_REVISION)
    parser.add_argument(
        "--pretrain-path",
        default=DEFAULT_PRETRAIN_PATH,
    )
    parser.add_argument("--pretrain-revision", default=DEFAULT_PRETRAIN_REVISION)
    parser.add_argument("--cpu-offload", choices=["none", "model", "sequential"], default="none")
    parser.add_argument(
        "--attention-backend",
        choices=["auto", "pytorch", "xformers"],
        default="auto",
        help="Attention implementation; PyTorch avoids optional CUDA extension coupling",
    )
    parser.add_argument("--process-length", type=int, default=-1)
    parser.add_argument("--target-fps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window-size", type=int, default=14)
    parser.add_argument("--time-step-size", type=int, default=10)
    parser.add_argument("--decode-chunk-size", type=int, default=7)
    parser.add_argument(
        "--video-decode-batch-size",
        type=int,
        default=16,
        help="Decode at most this many RGB frames into a temporary NumPy batch",
    )
    parser.add_argument(
        "--prefetch-next-video",
        action="store_true",
        help="Decode the next complete video in the background; disabled by default",
    )
    parser.add_argument(
        "--max-res",
        type=int,
        default=1024,
        help="Longest model-input side; 1024 maps 1280x720 input to 1024x576",
    )
    parser.add_argument("--output-width", type=int, default=1280)
    parser.add_argument("--output-height", type=int, default=720)
    parser.add_argument("--crf", type=int, default=17)
    parser.add_argument("--ffmpeg-preset", default="veryfast")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_shards < 1:
        raise ValueError("--num-shards must be at least 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= index < num_shards")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.episode_start is not None and args.episode_start < 0:
        raise ValueError("--episode-start must be non-negative")
    if args.episode_end is not None and args.episode_end < 0:
        raise ValueError("--episode-end must be non-negative")
    if (
        args.episode_start is not None
        and args.episode_end is not None
        and args.episode_start > args.episode_end
    ):
        raise ValueError("--episode-start cannot exceed --episode-end")
    for name in (
        "window_size",
        "time_step_size",
        "decode_chunk_size",
        "video_decode_batch_size",
        "max_res",
        "output_width",
        "output_height",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.time_step_size > args.window_size:
        raise ValueError("--time-step-size cannot exceed --window-size")
    if args.output_width % 2 or args.output_height % 2:
        raise ValueError("H.264 yuv420p output dimensions must be even")
    if not 0 <= args.crf <= 51:
        raise ValueError("--crf must be in [0, 51]")
    if args.stale_lock_hours < 0:
        raise ValueError("--stale-lock-hours cannot be negative")
    if args.lock_heartbeat_seconds < 0:
        raise ValueError("--lock-heartbeat-seconds cannot be negative")
    if args.max_attempts < 1:
        raise ValueError("--max-attempts must be at least 1")
    if args.retry_delay_seconds < 0:
        raise ValueError("--retry-delay-seconds cannot be negative")


def release_host_memory() -> None:
    """Collect Python objects and return free glibc heap pages to the host."""
    gc.collect()
    if not sys.platform.startswith("linux"):
        return
    try:
        libc = ctypes.CDLL(None)
        malloc_trim = libc.malloc_trim
    except (AttributeError, OSError):
        return
    malloc_trim.argtypes = [ctypes.c_size_t]
    malloc_trim.restype = ctypes.c_int
    malloc_trim(0)


def _warn_page_cache_once(key: str, message: str) -> None:
    """Report unsupported cache advice once without failing annotation work."""
    with _PAGE_CACHE_WARNING_LOCK:
        if key in _PAGE_CACHE_WARNINGS:
            return
        _PAGE_CACHE_WARNINGS.add(key)
    print(f"WARNING: {message}", file=sys.stderr, flush=True)


def release_file_page_cache(path: Path, *, sync_before_release: bool = False) -> bool:
    """Best-effort release of one finished file's Linux page-cache residency.

    This never changes or removes the file. Output files are synchronized before
    advice so their dirty pages become eligible for eviction. Unsupported file
    systems and permission errors are warnings rather than annotation failures.
    """
    if not sys.platform.startswith("linux"):
        return False
    posix_fadvise = getattr(os, "posix_fadvise", None)
    dontneed = getattr(os, "POSIX_FADV_DONTNEED", None)
    if posix_fadvise is None or dontneed is None:
        _warn_page_cache_once(
            "unsupported",
            "POSIX_FADV_DONTNEED is unavailable; file page cache will rely on kernel reclaim.",
        )
        return False

    flags = (os.O_RDWR if sync_before_release else os.O_RDONLY) | getattr(
        os, "O_CLOEXEC", 0
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        _warn_page_cache_once(
            f"open:{exc.errno}",
            f"cannot open a completed file for page-cache release ({exc}); continuing.",
        )
        return False

    try:
        if sync_before_release:
            try:
                os.fsync(descriptor)
            except OSError as exc:
                _warn_page_cache_once(
                    f"fsync:{exc.errno}",
                    f"cannot synchronize an output before page-cache release ({exc}); continuing.",
                )
        try:
            posix_fadvise(descriptor, 0, 0, dontneed)
        except OSError as exc:
            _warn_page_cache_once(
                f"fadvise:{exc.errno}",
                f"the filesystem rejected POSIX_FADV_DONTNEED ({exc}); continuing.",
            )
            return False
        return True
    finally:
        os.close(descriptor)


def discover_subset_roots(dataset_root: Path) -> list[Path]:
    roots: set[Path] = set()
    if (dataset_root / "videos").is_dir():
        roots.add(dataset_root)
    for child in dataset_root.iterdir():
        if not child.is_dir():
            continue
        if (child / "videos").is_dir():
            roots.add(child)
        for grandchild in child.iterdir():
            if grandchild.is_dir() and (grandchild / "videos").is_dir():
                roots.add(grandchild)
    return sorted(roots, key=lambda path: path.as_posix())


def normalized_episode_selectors(values: list[str] | None) -> set[str]:
    selected: set[str] = set()
    for value in values or []:
        if value.startswith("episode_"):
            selected.add(value)
        elif value.isdigit():
            selected.add(f"episode_{int(value):06d}")
        else:
            raise ValueError(f"Invalid episode selector: {value}")
    return selected


def normalized_chunk_selectors(values: list[str] | None) -> set[str]:
    selected: set[int] = set()
    for raw_value in values or []:
        for value in raw_value.split(","):
            value = value.strip()
            if not value:
                continue
            if value.startswith("chunk-"):
                suffix = value.removeprefix("chunk-")
                if not suffix.isdigit():
                    raise ValueError(f"Invalid chunk selector: {value}")
                selected.add(int(suffix))
                continue
            match = re.fullmatch(r"(\d+)(?:-(\d+))?", value)
            if match is None:
                raise ValueError(f"Invalid chunk selector: {value}")
            start = int(match.group(1))
            end = int(match.group(2) or start)
            if start > end:
                raise ValueError(f"Invalid descending chunk range: {value}")
            selected.update(range(start, end + 1))
    return {f"chunk-{chunk:03d}" for chunk in selected}


def camera_matches(camera: str, selected: set[str]) -> bool:
    if not selected:
        return True
    rgb_name = camera.removeprefix("observation.images.")
    suffix = rgb_name.removeprefix("rgb_")
    return bool({camera, rgb_name, suffix} & selected)


def discover_tasks(args: argparse.Namespace) -> list[NormalTask]:
    dataset_root = args.dataset_root.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    output_root = (args.output_root or dataset_root).resolve()
    log_root = (args.log_root or (output_root / "normalcrafter_logs")).resolve()

    subset_roots = discover_subset_roots(dataset_root)
    selected_subsets = set(args.subsets or [])
    if selected_subsets:
        subset_roots = [
            path
            for path in subset_roots
            if path.name in selected_subsets
            or path.relative_to(dataset_root).as_posix() in selected_subsets
        ]
        found = {
            key
            for path in subset_roots
            for key in (path.name, path.relative_to(dataset_root).as_posix())
        }
        missing = selected_subsets - found
        if missing:
            raise FileNotFoundError(f"Subsets not found: {', '.join(sorted(missing))}")

    selected_chunks = normalized_chunk_selectors(args.chunks)
    selected_cameras = set(args.cameras or [])
    selected_episodes = normalized_episode_selectors(args.episodes)
    tasks: list[NormalTask] = []

    for subset_root in subset_roots:
        relative = subset_root.relative_to(dataset_root)
        target_subset = output_root / relative
        for chunk_dir in sorted((subset_root / "videos").glob("chunk-*")):
            if selected_chunks and chunk_dir.name not in selected_chunks:
                continue
            for camera_dir in sorted(chunk_dir.glob("observation.images.rgb_*")):
                if not camera_matches(camera_dir.name, selected_cameras):
                    continue
                output_camera = camera_dir.name.replace(
                    "observation.images.rgb_", "observation.images.normal_", 1
                )
                for input_path in sorted(camera_dir.glob("episode_*.mp4")):
                    episode = input_path.stem
                    if selected_episodes and episode not in selected_episodes:
                        continue
                    try:
                        episode_number = int(episode.removeprefix("episode_"))
                    except ValueError:
                        continue
                    if args.episode_start is not None and episode_number < args.episode_start:
                        continue
                    if args.episode_end is not None and episode_number > args.episode_end:
                        continue
                    output_path = (
                        target_subset / "videos" / chunk_dir.name / output_camera / input_path.name
                    )
                    metadata_path = (
                        log_root / relative / chunk_dir.name / output_camera / f"{episode}.json"
                    )
                    tasks.append(
                        NormalTask(
                            subset_root=subset_root,
                            subset_relative=relative,
                            chunk=chunk_dir.name,
                            input_camera=camera_dir.name,
                            output_camera=output_camera,
                            episode=episode,
                            input_path=input_path,
                            output_path=output_path,
                            metadata_path=metadata_path,
                        )
                    )

    tasks.sort(
        key=lambda item: (
            item.subset_relative.as_posix(),
            item.chunk,
            item.input_camera,
            item.episode,
        )
    )
    return tasks


def annotation_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "version": ANNOTATION_CONFIG_VERSION,
        "method": "NormalCrafter",
        "normalcrafter_commit": NORMALCRAFTER_COMMIT,
        "unet": {"path": args.unet_path, "revision": args.unet_revision},
        "pretrain": {"path": args.pretrain_path, "revision": args.pretrain_revision},
        "settings": {
            "target_fps": args.target_fps,
            "process_length": args.process_length,
            "max_res": args.max_res,
            "window_size": args.window_size,
            "time_step_size": args.time_step_size,
            "decode_chunk_size": args.decode_chunk_size,
            "seed": args.seed,
            "cpu_offload": args.cpu_offload,
            "output_width": args.output_width,
            "output_height": args.output_height,
            "crf": args.crf,
            "ffmpeg_preset": args.ffmpeg_preset,
            "pixel_format": "yuv420p",
        },
    }


def config_fingerprint(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def has_mp4_signature(path: Path) -> bool:
    try:
        with path.open("rb") as file:
            header = file.read(12)
    except OSError:
        return False
    return len(header) == 12 and header[4:8] == b"ftyp"


def legacy_config_matches(payload: dict[str, Any], expected: dict[str, Any]) -> bool:
    """Accept compatible pre-fingerprint outputs without forcing a full relabel."""
    if payload.get("method") != expected["method"]:
        return False
    if payload.get("normalcrafter_commit") != expected["normalcrafter_commit"]:
        return False
    stored_settings = payload.get("settings")
    output = payload.get("output_video")
    if not isinstance(stored_settings, dict) or not isinstance(output, dict):
        return False
    expected_settings = expected["settings"]
    comparable = (
        "target_fps",
        "process_length",
        "max_res",
        "window_size",
        "time_step_size",
        "decode_chunk_size",
        "seed",
        "cpu_offload",
        "crf",
        "pixel_format",
    )
    if any(stored_settings.get(key) != expected_settings[key] for key in comparable):
        return False
    return (
        int(output.get("width", -1)) == expected_settings["output_width"]
        and int(output.get("height", -1)) == expected_settings["output_height"]
    )


def task_is_complete(task: NormalTask, expected_config: dict[str, Any] | None = None) -> bool:
    try:
        if not task.output_path.is_file() or not has_mp4_signature(task.output_path):
            return False
        output_size = task.output_path.stat().st_size
        payload = json.loads(task.metadata_path.read_text(encoding="utf-8"))
        output = payload["output_video"]
        if not (
            payload.get("status") == "complete"
            and int(output["width"]) > 0
            and int(output["height"]) > 0
            and int(output["frames"]) > 0
        ):
            return False
        recorded_output_size = output.get("size_bytes")
        if recorded_output_size is not None and int(recorded_output_size) != output_size:
            return False
        recorded_task = payload.get("task")
        recorded_input_size = (
            recorded_task.get("input_size_bytes") if isinstance(recorded_task, dict) else None
        )
        if (
            recorded_input_size is not None
            and int(recorded_input_size) != task.input_path.stat().st_size
        ):
            return False
        if expected_config is None:
            return True
        expected_fingerprint = config_fingerprint(expected_config)
        recorded_fingerprint = payload.get("config_fingerprint")
        if recorded_fingerprint is not None:
            return recorded_fingerprint == expected_fingerprint
        return legacy_config_matches(payload, expected_config)
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def select_worker_tasks(
    discovered: list[NormalTask],
    args: argparse.Namespace,
    expected_config: dict[str, Any] | None = None,
) -> tuple[list[NormalTask], int]:
    # Shard the stable discovered list before checking completion.  Otherwise two
    # machines that start at different times can see different pending indexes and
    # silently leave tasks unassigned.
    assigned = [
        task
        for index, task in enumerate(discovered)
        if index % args.num_shards == args.shard_index
    ]
    pending_count = sum(
        args.overwrite or not task_is_complete(task, expected_config) for task in discovered
    )
    worker_tasks = [
        task
        for task in assigned
        if args.overwrite or not task_is_complete(task, expected_config)
    ]
    if args.limit is not None:
        worker_tasks = worker_tasks[: args.limit]
    return worker_tasks, pending_count


def probe_video(path: Path) -> VideoInfo:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_packets",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_read_packets",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(result.stdout).get("streams", [])
    if len(streams) != 1:
        raise RuntimeError(f"Expected one video stream in {path}, found {len(streams)}")
    stream = streams[0]
    return VideoInfo(
        width=int(stream["width"]),
        height=int(stream["height"]),
        frames=int(stream["nb_read_packets"]),
        fps=str(stream["avg_frame_rate"]),
    )


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{path.name}.{os.getpid()}.part")
    with part.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(part, path)


def linux_boot_id() -> str | None:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
    except OSError:
        return None


def linux_process_start_ticks(pid: int) -> str | None:
    try:
        value = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return None
    closing_parenthesis = value.rfind(")")
    if closing_parenthesis < 0:
        return None
    fields_after_command = value[closing_parenthesis + 2 :].split()
    # /proc/<pid>/stat field 3 starts at index 0 here; process start time is field 22.
    return fields_after_command[19] if len(fields_after_command) > 19 else None


def process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class OutputLock:
    def __init__(
        self,
        output: Path,
        stale_hours: float,
        heartbeat_seconds: float = 30.0,
    ) -> None:
        self.path = output.with_name(f".{output.name}.lock")
        self.stale_seconds = stale_hours * 3600.0
        self.heartbeat_seconds = heartbeat_seconds
        self.hostname = socket.gethostname()
        self.boot_id = linux_boot_id()
        self.process_start_ticks = linux_process_start_ticks(os.getpid())
        self.token = uuid.uuid4().hex
        self.acquired = False
        self.recovered_stale_lock = False
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None

    def _payload(self) -> dict[str, Any]:
        return {
            "host": self.hostname,
            "pid": os.getpid(),
            "boot_id": self.boot_id,
            "process_start_ticks": self.process_start_ticks,
            "token": self.token,
            "created_unix": time.time(),
        }

    def _read_payload(self) -> dict[str, Any] | None:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    def _same_host_owner_is_alive(self, payload: dict[str, Any] | None) -> bool | None:
        if payload is None or payload.get("host") != self.hostname:
            return None
        if self.boot_id and payload.get("boot_id") not in (None, self.boot_id):
            return False
        try:
            pid = int(payload["pid"])
        except (KeyError, TypeError, ValueError):
            return None
        if not process_exists(pid):
            return False
        expected_start = payload.get("process_start_ticks")
        if expected_start is not None:
            current_start = linux_process_start_ticks(pid)
            if current_start is None:
                return False
            return str(expected_start) == current_start
        return True

    def _is_stale(self, payload: dict[str, Any] | None, age: float) -> bool:
        owner_alive = self._same_host_owner_is_alive(payload)
        if owner_alive is not None:
            return not owner_alive
        return bool(self.stale_seconds and age > self.stale_seconds)

    def _owns_lock(self) -> bool:
        payload = self._read_payload()
        return payload is not None and payload.get("token") == self.token

    def _heartbeat(self) -> None:
        while not self._heartbeat_stop.wait(self.heartbeat_seconds):
            if not self._owns_lock():
                return
            try:
                os.utime(self.path, None)
            except FileNotFoundError:
                return

    def _start_heartbeat(self) -> None:
        if self.heartbeat_seconds <= 0:
            return
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat,
            name=f"lock-heartbeat-{self.path.stem}",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(4):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                try:
                    snapshot = self.path.read_bytes()
                    stat = self.path.stat()
                except FileNotFoundError:
                    continue
                try:
                    payload = json.loads(snapshot)
                    if not isinstance(payload, dict):
                        payload = None
                except json.JSONDecodeError:
                    payload = None
                age = max(0.0, time.time() - stat.st_mtime)
                if not self._is_stale(payload, age):
                    return False
                try:
                    if self.path.read_bytes() != snapshot:
                        continue
                    self.path.unlink()
                except FileNotFoundError:
                    pass
                self.recovered_stale_lock = True
                continue
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as file:
                    json.dump(self._payload(), file, sort_keys=True)
                    file.write("\n")
                    file.flush()
                    os.fsync(file.fileno())
            except BaseException:
                self.path.unlink(missing_ok=True)
                raise
            self.acquired = True
            self._start_heartbeat()
            return True
        return False

    def release(self) -> None:
        if not self.acquired:
            return
        self._heartbeat_stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join()
        if self._owns_lock():
            self.path.unlink(missing_ok=True)
        self.acquired = False


def cleanup_orphan_task_parts(task: NormalTask) -> tuple[int, int]:
    candidates: set[Path] = set(
        task.output_path.parent.glob(f".{task.output_path.stem}.*.part.mp4")
    )
    failed_metadata_path = task.metadata_path.with_suffix(".failed.json")
    for metadata_path in (task.metadata_path, failed_metadata_path):
        candidates.update(metadata_path.parent.glob(f".{metadata_path.name}.*.part"))
    removed = 0
    removed_bytes = 0
    for path in candidates:
        try:
            removed_bytes += path.stat().st_size
            path.unlink()
            removed += 1
        except FileNotFoundError:
            continue
    return removed, removed_bytes


class AsyncMp4Writer:
    def __init__(
        self,
        path: Path,
        *,
        width: int,
        height: int,
        fps: float,
        expected_frames: int,
        crf: int,
        preset: str,
        queue_size: int = 3,
    ) -> None:
        self.path = path
        safe_host = re.sub(r"[^A-Za-z0-9_.-]+", "_", socket.gethostname())
        self.tmp_path = path.with_name(
            f".{path.stem}.{safe_host}.{os.getpid()}.{uuid.uuid4().hex}.part.mp4"
        )
        self.width = width
        self.height = height
        self.fps = fps
        self.expected_frames = expected_frames
        self.crf = crf
        self.preset = preset
        self.queue: queue.Queue[np.ndarray | object] = queue.Queue(queue_size)
        self.error: BaseException | None = None
        self.written_frames = 0
        self.thread = threading.Thread(target=self._run, name=f"encode-{path.stem}")
        self.thread.start()

    def _put(self, item: np.ndarray | object) -> None:
        while True:
            if self.error is not None:
                raise RuntimeError(f"MP4 encoder failed for {self.path}") from self.error
            try:
                self.queue.put(item, timeout=0.5)
                return
            except queue.Full:
                continue

    def write(self, frames: np.ndarray) -> None:
        if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"Expected uint8 TxHxWx3 frames, got {frames.shape} {frames.dtype}")
        if frames.shape[1:3] != (self.height, self.width):
            raise ValueError(
                f"Frame size changed: {frames.shape[2]}x{frames.shape[1]} != "
                f"{self.width}x{self.height}"
            )
        self._put(frames)

    def finish(self) -> None:
        self._put(_STOP)
        self.thread.join()
        if self.error is not None:
            raise RuntimeError(f"MP4 encoder failed for {self.path}") from self.error

    def abort(self) -> None:
        if self.thread.is_alive():
            try:
                self._put(_ABORT)
            except RuntimeError:
                pass
            self.thread.join()

    def _run(self) -> None:
        process: subprocess.Popen[bytes] | None = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.tmp_path.unlink(missing_ok=True)
            process = subprocess.Popen(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-nostdin",
                    "-y",
                    "-f",
                    "rawvideo",
                    "-vcodec",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-s",
                    f"{self.width}x{self.height}",
                    "-r",
                    str(self.fps),
                    "-i",
                    "-",
                    "-an",
                    "-vcodec",
                    "libx264",
                    "-preset",
                    self.preset,
                    "-threads",
                    "1",
                    "-crf",
                    str(self.crf),
                    "-pix_fmt",
                    "yuv420p",
                    str(self.tmp_path),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            if process.stdin is None:
                raise RuntimeError("ffmpeg stdin is unavailable")

            aborted = False
            while True:
                item = self.queue.get()
                if item is _STOP:
                    break
                if item is _ABORT:
                    aborted = True
                    break
                assert isinstance(item, np.ndarray)
                process.stdin.write(item.tobytes())
                self.written_frames += len(item)

            if aborted:
                process.kill()
                process.wait()
                self.tmp_path.unlink(missing_ok=True)
                return

            process.stdin.close()
            stderr = process.stderr.read().decode(errors="replace") if process.stderr else ""
            return_code = process.wait()
            if return_code:
                raise RuntimeError(stderr.strip() or f"ffmpeg exited with code {return_code}")
            if self.written_frames != self.expected_frames:
                raise RuntimeError(
                    f"Encoded frame count mismatch: {self.written_frames} != {self.expected_frames}"
                )
            info = probe_video(self.tmp_path)
            if (info.width, info.height, info.frames) != (
                self.width,
                self.height,
                self.expected_frames,
            ):
                raise RuntimeError(
                    f"Invalid encoded video: {info.width}x{info.height}, {info.frames} frames"
                )
            os.replace(self.tmp_path, self.path)
        except BaseException as exc:
            self.error = exc
            if process is not None and process.poll() is None:
                process.kill()
                process.wait()
            self.tmp_path.unlink(missing_ok=True)


class NormalCrafterRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        root = args.normalcrafter_root.resolve()
        pipeline_path = root / "normalcrafter" / "normal_crafter_ppl.py"
        if not pipeline_path.is_file():
            raise FileNotFoundError(f"NormalCrafter checkout is incomplete: {root}")
        sys.path.insert(0, str(root))

        self.verbose_inference = bool(getattr(args, "verbose_inference", False))
        requested_attention = getattr(args, "attention_backend", "auto")
        with ExitStack() as stack:
            if not self.verbose_inference:
                devnull = stack.enter_context(open(os.devnull, "w", encoding="utf-8"))
                stack.enter_context(redirect_stdout(devnull))
                stack.enter_context(redirect_stderr(devnull))

            import inspect
            import torch
            from diffusers import AutoencoderKLTemporalDecoder
            from diffusers.training_utils import set_seed
            from normalcrafter.normal_crafter_ppl import NormalCrafterPipeline
            from normalcrafter.unet import DiffusersUNetSpatioTemporalConditionModelNormalCrafter
            from normalcrafter.utils import read_video_frames

            if requested_attention == "auto":
                gpu_name = torch.cuda.get_device_name(0)
                self.attention_backend = (
                    "pytorch"
                    if re.search(r"H100|H200|B100|B200", gpu_name, re.IGNORECASE)
                    else "xformers"
                )
            else:
                self.attention_backend = requested_attention

            if "output_type" not in inspect.signature(NormalCrafterPipeline.__call__).parameters:
                raise RuntimeError(
                    "NormalCrafter long-video patch is not applied. "
                    "See droid_normals/NORMALCRAFTER.md."
                )
            if "chunk_size" not in inspect.signature(
                NormalCrafterPipeline._encode_image
            ).parameters:
                raise RuntimeError(
                    "NormalCrafter bounded-input patch is not applied. "
                    "Run install_normalcrafter.sh again."
                )

            self.torch = torch
            self.inference_dtype = torch.float16
            self.set_seed = set_seed
            self.read_video_frames = read_video_frames
            load_started = time.monotonic()
            unet = DiffusersUNetSpatioTemporalConditionModelNormalCrafter.from_pretrained(
                args.unet_path,
                subfolder="unet",
                revision=args.unet_revision,
                low_cpu_mem_usage=True,
            )
            vae = AutoencoderKLTemporalDecoder.from_pretrained(
                args.unet_path,
                subfolder="vae",
                revision=args.unet_revision,
                low_cpu_mem_usage=True,
            )
            vae.to(dtype=torch.float16)
            unet.to(dtype=torch.float16)
            self.pipe = NormalCrafterPipeline.from_pretrained(
                args.pretrain_path,
                unet=unet,
                vae=vae,
                torch_dtype=torch.float16,
                variant="fp16",
                revision=args.pretrain_revision,
                low_cpu_mem_usage=True,
            )
            self.pipe.set_progress_bar_config(disable=not self.verbose_inference)
            if args.cpu_offload == "none":
                self.pipe.to("cuda")
            elif args.cpu_offload == "model":
                self.pipe.enable_model_cpu_offload()
            else:
                self.pipe.enable_sequential_cpu_offload()
            self.load_seconds = time.monotonic() - load_started
        if self.attention_backend == "xformers":
            self.pipe.enable_xformers_memory_efficient_attention()
        print(f"Attention backend: {self.attention_backend}", flush=True)

    def restore_model_dtypes(self) -> list[str]:
        restored: list[str] = []
        for name in ("vae", "unet"):
            module = getattr(self.pipe, name)
            if module.dtype != self.inference_dtype:
                module.to(dtype=self.inference_dtype)
                restored.append(name)
        return restored

    def reset_after_attempt(self, *, clear_cuda_cache: bool = False) -> list[str]:
        """Release task allocations and clear CUDA cache only after abnormal attempts."""
        torch = self.torch
        needs_dtype_restore = any(
            getattr(self.pipe, name).dtype != self.inference_dtype for name in ("vae", "unet")
        )
        if clear_cuda_cache or needs_dtype_restore:
            torch.cuda.empty_cache()
        restored = self.restore_model_dtypes()
        if restored:
            torch.cuda.empty_cache()
        release_host_memory()
        return restored

    def load_video(self, task: NormalTask, args: argparse.Namespace) -> tuple[list[Any], float]:
        if self.verbose_inference:
            return self.read_video_frames(
                str(task.input_path),
                args.process_length,
                args.target_fps,
                args.max_res,
            )

        from decord import VideoReader, cpu
        from PIL import Image

        probe = VideoReader(str(task.input_path), ctx=cpu(0))
        try:
            original_height, original_width = probe.get_batch([0]).shape[1:3]
        finally:
            del probe
        if max(original_height, original_width) > args.max_res:
            scale = args.max_res / max(original_height, original_width)
            height = round(original_height * scale)
            width = round(original_width * scale)
        else:
            height = original_height
            width = original_width
        video = VideoReader(
            str(task.input_path),
            ctx=cpu(0),
            width=width,
            height=height,
        )
        fps = video.get_avg_fps() if args.target_fps == -1 else args.target_fps
        stride = max(round(video.get_avg_fps() / fps), 1)
        frame_indexes = list(range(0, len(video), stride))
        if args.process_length != -1:
            frame_indexes = frame_indexes[: args.process_length]
        frames: list[Any] = []
        try:
            for start in range(0, len(frame_indexes), args.video_decode_batch_size):
                indexes = frame_indexes[start : start + args.video_decode_batch_size]
                batch = video.get_batch(indexes).asnumpy()
                if batch.dtype != np.uint8:
                    batch = batch.astype(np.uint8)
                for frame in batch:
                    source = Image.fromarray(np.ascontiguousarray(frame))
                    try:
                        # Detach each PIL image from the temporary Decord/NumPy batch.
                        frames.append(source.copy())
                    finally:
                        source.close()
                del batch
        finally:
            del video
            release_host_memory()
        return frames, float(fps)

    def infer(
        self,
        frames: list[Any],
        writer: AsyncMp4Writer,
        args: argparse.Namespace,
    ) -> dict[str, Any]:
        torch = self.torch
        import torch.nn.functional as functional

        restored = self.restore_model_dtypes()
        if restored:
            print(f"RECOVERED model dtype before inference: {','.join(restored)}", flush=True)
        self.set_seed(args.seed)
        input_width, input_height = frames[0].size
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.monotonic()
        with torch.inference_mode():
            latents = self.pipe(
                frames,
                decode_chunk_size=args.decode_chunk_size,
                time_step_size=args.time_step_size,
                window_size=args.window_size,
                output_type="latent",
            ).frames

            for start in range(0, len(frames), args.decode_chunk_size):
                end = min(start + args.decode_chunk_size, len(frames))
                decoded = self.pipe.decode_latents(
                    latents[:, start:end],
                    end - start,
                    decode_chunk_size=end - start,
                )
                normals = self.pipe.video_processor.postprocess_video(
                    decoded, output_type="pt"
                )[0]
                normals = normals.mul_(2.0).sub_(1.0)

                pad_height = normals.shape[-2] - input_height
                pad_width = normals.shape[-1] - input_width
                top = pad_height // 2
                left = pad_width // 2
                normals = normals[
                    :, :, top : top + input_height, left : left + input_width
                ]
                if normals.shape[-2:] != (writer.height, writer.width):
                    normals = functional.interpolate(
                        normals,
                        size=(writer.height, writer.width),
                        mode="bilinear",
                        align_corners=False,
                    )
                normals = functional.normalize(normals, p=2, dim=1, eps=1e-6)
                encoded = (
                    normals.mul(0.5)
                    .add(0.5)
                    .mul(255.0)
                    .clamp_(0.0, 255.0)
                    .to(torch.uint8)
                    .permute(0, 2, 3, 1)
                    .contiguous()
                    .cpu()
                    .numpy()
                )
                writer.write(encoded)
                del decoded, normals, encoded
        del latents
        torch.cuda.synchronize()
        return {
            "model_input_width": input_width,
            "model_input_height": input_height,
            "gpu_work_seconds": time.monotonic() - started,
            "max_cuda_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        }


def task_payload(task: NormalTask) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "subset": task.subset_relative.as_posix(),
        "chunk": task.chunk,
        "input_camera": task.input_camera,
        "output_camera": task.output_camera,
        "episode": task.episode,
        "input_path": str(task.input_path),
        "output_path": str(task.output_path),
    }
    try:
        payload["input_size_bytes"] = task.input_path.stat().st_size
    except OSError:
        payload["input_size_bytes"] = None
    return payload


def process_task(
    runner: NormalCrafterRunner,
    task: NormalTask,
    frames: list[Any],
    fps: float,
    args: argparse.Namespace,
    expected_config: dict[str, Any],
    attempt: int,
) -> str:
    lock = OutputLock(
        task.output_path,
        args.stale_lock_hours,
        args.lock_heartbeat_seconds,
    )
    if not lock.acquire():
        print(f"LOCKED: {task.output_path}", flush=True)
        return "locked"

    writer: AsyncMp4Writer | None = None
    started = time.monotonic()
    try:
        removed_parts, removed_bytes = cleanup_orphan_task_parts(task)
        if lock.recovered_stale_lock or removed_parts:
            print(
                f"RECOVERED: {task.output_path} | "
                f"stale_lock={int(lock.recovered_stale_lock)} | "
                f"orphan_parts={removed_parts} ({removed_bytes} bytes)",
                flush=True,
            )
        if task_is_complete(task, expected_config) and not args.overwrite:
            print(f"SKIP: {task.output_path}", flush=True)
            return "skipped"
        input_info = probe_video(task.input_path)
        writer = AsyncMp4Writer(
            task.output_path,
            width=args.output_width,
            height=args.output_height,
            fps=fps,
            expected_frames=len(frames),
            crf=args.crf,
            preset=args.ffmpeg_preset,
        )
        inference = runner.infer(frames, writer, args)
        writer.finish()
        writer = None
        output_info = probe_video(task.output_path)
        output_video = {
            **vars(output_info),
            "size_bytes": task.output_path.stat().st_size,
        }
        output_cache_released = release_file_page_cache(
            task.output_path,
            sync_before_release=True,
        )
        payload: dict[str, Any] = {
            "status": "complete",
            "method": "NormalCrafter",
            "normalcrafter_commit": NORMALCRAFTER_COMMIT,
            "normal_convention": "native NormalCrafter view space; RGB=(normal+1)/2",
            "annotation_config": expected_config,
            "config_fingerprint": config_fingerprint(expected_config),
            "task": task_payload(task),
            "input_video": vars(input_info),
            "output_video": output_video,
            "settings": {
                "target_fps": args.target_fps,
                "process_length": args.process_length,
                "max_res": args.max_res,
                "window_size": args.window_size,
                "time_step_size": args.time_step_size,
                "decode_chunk_size": args.decode_chunk_size,
                "seed": args.seed,
                "cpu_offload": args.cpu_offload,
                "attention_backend": runner.attention_backend,
                "crf": args.crf,
                "pixel_format": "yuv420p",
                "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
                "file_page_cache_release": {
                    "input": "best_effort_after_attempt",
                    "output_released": output_cache_released,
                },
            },
            "model_load_seconds": runner.load_seconds,
            "attempt": attempt,
            **inference,
            "wall_seconds": time.monotonic() - started,
            "host": socket.gethostname(),
        }
        write_json_atomic(task.metadata_path, payload)
        task.metadata_path.with_suffix(".failed.json").unlink(missing_ok=True)
        print(
            f"SAVED: {task.output_path} | {len(frames)} frames | "
            f"{payload['wall_seconds']:.1f}s",
            flush=True,
        )
        return "complete"
    except BaseException as exc:
        if writer is not None:
            writer.abort()
        failure = {
            "status": "failed",
            "method": "NormalCrafter",
            "task": task_payload(task),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "host": socket.gethostname(),
            "wall_seconds": time.monotonic() - started,
            "attempt": attempt,
            "max_attempts": args.max_attempts,
        }
        write_json_atomic(task.metadata_path.with_suffix(".failed.json"), failure)
        raise
    finally:
        lock.release()


def print_plan(
    tasks: list[NormalTask],
    args: argparse.Namespace,
    discovered_count: int,
    pending_count: int,
) -> None:
    print("NormalCrafter annotation plan")
    print(f"  Dataset root:     {args.dataset_root.resolve()}")
    print(f"  Output root:      {(args.output_root or args.dataset_root).resolve()}")
    print(f"  Shard:            {args.shard_index}/{args.num_shards}")
    print(f"  Discovered:       {discovered_count}")
    print(f"  Pending globally: {pending_count}")
    print(f"  Pending on shard: {len(tasks)}")
    print(f"  Model max side:   {args.max_res}")
    print(f"  Output:           {args.output_width}x{args.output_height} H.264 CRF {args.crf}")
    print(f"  RGB decode batch: {args.video_decode_batch_size} frame(s)")
    print(f"  Next-video fetch: {'on' if args.prefetch_next_video else 'off'}")
    print("  File page cache:  release input/output after each attempt (Linux best effort)")
    for task in tasks[: args.print_limit]:
        print(f"  {task.input_path} -> {task.output_path}")
    if len(tasks) > args.print_limit:
        print(f"  ... and {len(tasks) - args.print_limit} more")


def main() -> None:
    args = parse_args()
    validate_args(args)

    def handle_sigterm(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt("received SIGTERM")

    signal.signal(signal.SIGTERM, handle_sigterm)
    if not args.dry_run:
        for executable in ("ffmpeg", "ffprobe"):
            if not any(
                (Path(directory) / executable).is_file()
                for directory in os.environ.get("PATH", "").split(os.pathsep)
            ):
                raise RuntimeError(f"{executable} is required but was not found in PATH")

    expected_config = annotation_config(args)
    discovered = discover_tasks(args)
    tasks, pending_count = select_worker_tasks(discovered, args, expected_config)
    print_plan(tasks, args, len(discovered), pending_count)
    if args.dry_run:
        return
    if not tasks:
        print("All selected normal videos already exist.", flush=True)
        return

    import torch

    torch.backends.cudnn.benchmark = False
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    print(f"Loading NormalCrafter once for {len(tasks)} video(s)...", flush=True)
    runner = NormalCrafterRunner(args)
    release_host_memory()
    print(f"NormalCrafter loaded in {runner.load_seconds:.1f}s.", flush=True)
    succeeded = 0
    failed = 0
    skipped = 0
    locked = 0

    pool: ThreadPoolExecutor | None = None
    future: Future[tuple[list[Any], float]] | None = None
    if args.prefetch_next_video:
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="video-prefetch")
        future = pool.submit(runner.load_video, tasks[0], args)
    try:
        for index, task in enumerate(tasks):
            current_future = future
            if pool is not None and index + 1 < len(tasks):
                future = pool.submit(runner.load_video, tasks[index + 1], args)
            print(
                f"[{index + 1}/{len(tasks)}] {task.chunk} | "
                f"{task.input_camera} | {task.episode}",
                flush=True,
            )
            final_error: tuple[str, str] | None = None
            result: str | None = None
            for attempt in range(1, args.max_attempts + 1):
                frames: list[Any] | None = None
                attempt_failed = False
                retrying = False
                try:
                    if attempt == 1:
                        if current_future is None:
                            frames, fps = runner.load_video(task, args)
                        else:
                            frames, fps = current_future.result()
                    else:
                        frames, fps = runner.load_video(task, args)
                    result = process_task(
                        runner,
                        task,
                        frames,
                        float(fps),
                        args,
                        expected_config,
                        attempt,
                    )
                    final_error = None
                    break
                except Exception as exc:
                    attempt_failed = True
                    final_error = (type(exc).__name__, str(exc))
                    if attempt < args.max_attempts:
                        retrying = True
                        print(
                            f"RETRY {attempt + 1}/{args.max_attempts}: "
                            f"{task.input_path}: {exc}",
                            file=sys.stderr,
                            flush=True,
                        )
                    if attempt == args.max_attempts and not args.continue_on_error:
                        raise
                finally:
                    if frames is not None:
                        del frames
                    release_file_page_cache(task.input_path)
                    restored = runner.reset_after_attempt(clear_cuda_cache=attempt_failed)
                    if restored:
                        print(
                            f"RECOVERED model dtype after attempt: {','.join(restored)}",
                            flush=True,
                        )
                if retrying and args.retry_delay_seconds:
                    time.sleep(args.retry_delay_seconds)

            if final_error is not None:
                failed += 1
                error_type, error_message = final_error
                print(
                    f"FAILED after {args.max_attempts} attempt(s): "
                    f"{task.input_path}: {error_type}: {error_message}",
                    file=sys.stderr,
                    flush=True,
                )
            elif result == "complete":
                succeeded += 1
            elif result == "locked":
                locked += 1
            else:
                skipped += 1
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)

    remaining = [task for task in tasks if not task_is_complete(task, expected_config)]
    print(
        "NormalCrafter complete: "
        f"{succeeded} succeeded, {skipped} skipped, {locked} locked, {failed} failed, "
        f"{len(remaining)} still pending",
        flush=True,
    )
    if remaining:
        for task in remaining[: args.print_limit]:
            print(f"PENDING: {task.output_path}", file=sys.stderr, flush=True)
        if len(remaining) > args.print_limit:
            print(
                f"... and {len(remaining) - args.print_limit} more pending output(s)",
                file=sys.stderr,
                flush=True,
            )
    if failed or remaining:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
