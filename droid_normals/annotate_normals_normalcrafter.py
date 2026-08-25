#!/usr/bin/env python3
"""Annotate LeRobot RGB videos with NormalCrafter surface-normal MP4s."""

from __future__ import annotations

import argparse
import gc
import json
import os
import queue
import re
import socket
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


_STOP = object()
_ABORT = object()


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
        default=24.0,
        help="Replace output locks older than this many hours; 0 disables replacement",
    )

    parser.add_argument("--unet-path", default="Yanrui95/NormalCrafter")
    parser.add_argument(
        "--pretrain-path",
        default="stabilityai/stable-video-diffusion-img2vid-xt",
    )
    parser.add_argument("--cpu-offload", choices=["none", "model", "sequential"], default="none")
    parser.add_argument("--process-length", type=int, default=-1)
    parser.add_argument("--target-fps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window-size", type=int, default=14)
    parser.add_argument("--time-step-size", type=int, default=10)
    parser.add_argument("--decode-chunk-size", type=int, default=7)
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
    if args.max_attempts < 1:
        raise ValueError("--max-attempts must be at least 1")
    if args.retry_delay_seconds < 0:
        raise ValueError("--retry-delay-seconds cannot be negative")


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


def task_is_complete(task: NormalTask) -> bool:
    try:
        if not task.output_path.is_file() or task.output_path.stat().st_size <= 0:
            return False
        payload = json.loads(task.metadata_path.read_text(encoding="utf-8"))
        output = payload["output_video"]
        return (
            payload.get("status") == "complete"
            and int(output["width"]) > 0
            and int(output["height"]) > 0
            and int(output["frames"]) > 0
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def select_worker_tasks(
    discovered: list[NormalTask], args: argparse.Namespace
) -> tuple[list[NormalTask], int]:
    pending = [task for task in discovered if args.overwrite or not task_is_complete(task)]
    worker_tasks = [
        task
        for index, task in enumerate(pending)
        if index % args.num_shards == args.shard_index
    ]
    if args.limit is not None:
        worker_tasks = worker_tasks[: args.limit]
    return worker_tasks, len(pending)


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


class OutputLock:
    def __init__(self, output: Path, stale_hours: float) -> None:
        self.path = output.with_name(f".{output.name}.lock")
        self.stale_seconds = stale_hours * 3600.0
        self.acquired = False

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                if not self.stale_seconds:
                    return False
                try:
                    age = time.time() - self.path.stat().st_mtime
                except FileNotFoundError:
                    continue
                if age <= self.stale_seconds:
                    return False
                try:
                    self.path.unlink()
                except FileNotFoundError:
                    pass
                continue
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                json.dump(
                    {
                        "host": socket.gethostname(),
                        "pid": os.getpid(),
                        "created_unix": time.time(),
                    },
                    file,
                )
                file.write("\n")
            self.acquired = True
            return True
        return False

    def release(self) -> None:
        if self.acquired:
            self.path.unlink(missing_ok=True)
            self.acquired = False


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
        self.tmp_path = path.with_name(f".{path.stem}.{os.getpid()}.part.mp4")
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

        import inspect
        import torch
        from diffusers import AutoencoderKLTemporalDecoder
        from diffusers.training_utils import set_seed
        from normalcrafter.normal_crafter_ppl import NormalCrafterPipeline
        from normalcrafter.unet import DiffusersUNetSpatioTemporalConditionModelNormalCrafter
        from normalcrafter.utils import read_video_frames

        if "output_type" not in inspect.signature(NormalCrafterPipeline.__call__).parameters:
            raise RuntimeError(
                "NormalCrafter long-video patch is not applied. "
                "See droid_normals/NORMALCRAFTER.md."
            )

        self.torch = torch
        self.set_seed = set_seed
        self.read_video_frames = read_video_frames
        load_started = time.monotonic()
        unet = DiffusersUNetSpatioTemporalConditionModelNormalCrafter.from_pretrained(
            args.unet_path,
            subfolder="unet",
            low_cpu_mem_usage=True,
        )
        vae = AutoencoderKLTemporalDecoder.from_pretrained(args.unet_path, subfolder="vae")
        vae.to(dtype=torch.float16)
        unet.to(dtype=torch.float16)
        self.pipe = NormalCrafterPipeline.from_pretrained(
            args.pretrain_path,
            unet=unet,
            vae=vae,
            torch_dtype=torch.float16,
            variant="fp16",
        )
        if args.cpu_offload == "none":
            self.pipe.to("cuda")
        elif args.cpu_offload == "model":
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.enable_sequential_cpu_offload()
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except Exception as exc:
            print(f"WARNING: xFormers is unavailable: {exc}", flush=True)
        self.load_seconds = time.monotonic() - load_started

    def load_video(self, task: NormalTask, args: argparse.Namespace) -> tuple[list[Any], float]:
        return self.read_video_frames(
            str(task.input_path),
            args.process_length,
            args.target_fps,
            args.max_res,
        )

    def infer(
        self,
        frames: list[Any],
        writer: AsyncMp4Writer,
        args: argparse.Namespace,
    ) -> dict[str, Any]:
        torch = self.torch
        import torch.nn.functional as functional

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


def task_payload(task: NormalTask) -> dict[str, str]:
    return {
        "subset": task.subset_relative.as_posix(),
        "chunk": task.chunk,
        "input_camera": task.input_camera,
        "output_camera": task.output_camera,
        "episode": task.episode,
        "input_path": str(task.input_path),
        "output_path": str(task.output_path),
    }


def process_task(
    runner: NormalCrafterRunner,
    task: NormalTask,
    frames: list[Any],
    fps: float,
    args: argparse.Namespace,
    attempt: int,
) -> str:
    lock = OutputLock(task.output_path, args.stale_lock_hours)
    if not lock.acquire():
        print(f"LOCKED: {task.output_path}", flush=True)
        return "locked"

    writer: AsyncMp4Writer | None = None
    started = time.monotonic()
    try:
        if task_is_complete(task) and not args.overwrite:
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
        payload: dict[str, Any] = {
            "status": "complete",
            "method": "NormalCrafter",
            "normalcrafter_commit": "75af9887a2cb14cd1ce3883c5773bc296565777c",
            "normal_convention": "native NormalCrafter view space; RGB=(normal+1)/2",
            "task": task_payload(task),
            "input_video": vars(input_info),
            "output_video": vars(output_info),
            "settings": {
                "target_fps": args.target_fps,
                "process_length": args.process_length,
                "max_res": args.max_res,
                "window_size": args.window_size,
                "time_step_size": args.time_step_size,
                "decode_chunk_size": args.decode_chunk_size,
                "seed": args.seed,
                "cpu_offload": args.cpu_offload,
                "crf": args.crf,
                "pixel_format": "yuv420p",
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
    for task in tasks[: args.print_limit]:
        print(f"  {task.input_path} -> {task.output_path}")
    if len(tasks) > args.print_limit:
        print(f"  ... and {len(tasks) - args.print_limit} more")


def main() -> None:
    args = parse_args()
    validate_args(args)
    if not args.dry_run:
        for executable in ("ffmpeg", "ffprobe"):
            if not any(
                (Path(directory) / executable).is_file()
                for directory in os.environ.get("PATH", "").split(os.pathsep)
            ):
                raise RuntimeError(f"{executable} is required but was not found in PATH")

    discovered = discover_tasks(args)
    tasks, pending_count = select_worker_tasks(discovered, args)
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
    succeeded = 0
    failed = 0
    skipped = 0
    locked = 0

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="video-prefetch") as pool:
        future: Future[tuple[list[Any], float]] = pool.submit(runner.load_video, tasks[0], args)
        for index, task in enumerate(tasks):
            current_future = future
            if index + 1 < len(tasks):
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
                try:
                    if attempt == 1:
                        frames, fps = current_future.result()
                    else:
                        frames, fps = runner.load_video(task, args)
                    result = process_task(runner, task, frames, float(fps), args, attempt)
                    final_error = None
                    break
                except Exception as exc:
                    final_error = (type(exc).__name__, str(exc))
                    if attempt < args.max_attempts:
                        print(
                            f"RETRY {attempt + 1}/{args.max_attempts}: "
                            f"{task.input_path}: {exc}",
                            file=sys.stderr,
                            flush=True,
                        )
                    gc.collect()
                    torch.cuda.empty_cache()
                    if attempt < args.max_attempts and args.retry_delay_seconds:
                        time.sleep(args.retry_delay_seconds)
                    if attempt == args.max_attempts and not args.continue_on_error:
                        raise
                finally:
                    if frames is not None:
                        del frames

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

    print(
        "NormalCrafter complete: "
        f"{succeeded} succeeded, {skipped} skipped, {locked} locked, {failed} failed",
        flush=True,
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
