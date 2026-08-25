#!/usr/bin/env python3
"""Download selected LeRobot RGB-video chunks from a Hugging Face dataset."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def parse_chunks(value: str) -> list[int]:
    chunks: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        match = re.fullmatch(r"(?:chunk-)?(\d+)(?:-(\d+))?", item)
        if match is None:
            raise ValueError(f"Invalid chunk selector: {item}")
        start = int(match.group(1))
        end = int(match.group(2) or start)
        if start > end:
            raise ValueError(f"Invalid descending chunk range: {item}")
        chunks.update(range(start, end + 1))
    if not chunks:
        raise ValueError("No chunks were selected")
    return sorted(chunks)


def normalize_camera(value: str) -> str:
    if value.startswith("observation.images."):
        return value
    if value.startswith("rgb_"):
        return f"observation.images.{value}"
    if value.isdigit():
        return f"observation.images.rgb_{int(value):02d}"
    raise ValueError(f"Invalid camera selector: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_id")
    parser.add_argument("chunks")
    parser.add_argument("download_dir", type=Path)
    parser.add_argument("--cameras", nargs="+", default=["01", "02"])
    parser.add_argument("--revision")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chunks = parse_chunks(args.chunks)
    cameras = sorted({normalize_camera(camera) for camera in args.cameras})
    if args.workers < 1:
        raise ValueError("--workers must be positive")

    patterns = ["meta/**"]
    patterns.extend(
        f"videos/chunk-{chunk:03d}/{camera}/*.mp4"
        for chunk in chunks
        for camera in cameras
    )
    download_dir = args.download_dir.resolve()
    print(f"Repository:  {args.repo_id}")
    print(f"Revision:    {args.revision or 'main'}")
    print(f"Chunks:      {', '.join(f'{chunk:03d}' for chunk in chunks)}")
    print(f"Cameras:     {', '.join(cameras)}")
    print(f"Destination: {download_dir}")
    if args.dry_run:
        for pattern in patterns:
            print(f"  {pattern}")
        return

    from huggingface_hub import snapshot_download

    download_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=download_dir,
        allow_patterns=patterns,
        max_workers=args.workers,
    )

    missing: list[Path] = []
    video_count = 0
    total_bytes = 0
    for chunk in chunks:
        for camera in cameras:
            camera_dir = download_dir / "videos" / f"chunk-{chunk:03d}" / camera
            videos = sorted(camera_dir.glob("episode_*.mp4"))
            if not videos:
                missing.append(camera_dir)
            video_count += len(videos)
            total_bytes += sum(video.stat().st_size for video in videos)
    if missing:
        preview = "\n".join(str(path) for path in missing[:10])
        raise RuntimeError(
            f"No MP4 files were downloaded for {len(missing)} chunk/camera paths, including:\n"
            f"{preview}\nCheck the repository ID and camera names."
        )
    print(
        f"Download complete: {video_count} RGB videos, "
        f"{total_bytes / 2**30:.2f} GiB materialized."
    )


if __name__ == "__main__":
    main()
