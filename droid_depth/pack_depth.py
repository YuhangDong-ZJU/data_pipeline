#!/usr/bin/env python3
"""Pack LeRobot depth PNG directories into episode-aligned TAR shards."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Shard:
    chunk: str
    camera: str
    source_root: Path
    episodes: tuple[Path, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pack depth episode directories into uncompressed TAR shards."
    )
    parser.add_argument("dataset_root", type=Path, help="Root containing LeRobot subsets")
    parser.add_argument(
        "--subsets",
        nargs="+",
        help="Subset names to process; default: all subsets containing images/",
    )
    parser.add_argument(
        "--chunks",
        nargs="+",
        help="Chunk names to process, for example chunk-000 chunk-001; default: all",
    )
    parser.add_argument(
        "--episodes-per-shard",
        type=int,
        default=250,
        help="Maximum episode directories in each TAR (default: 250)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of TAR processes running at once (default: 1)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recreate TAR files that already exist",
    )
    parser.add_argument(
        "--delete-source",
        action="store_true",
        help="Delete source episode directories after their TAR is safely written",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the planned TAR files without writing them",
    )
    return parser.parse_args()


def discover_shards(args: argparse.Namespace) -> list[Shard]:
    dataset_root = args.dataset_root.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    subsets = [path for path in dataset_root.iterdir() if (path / "images").is_dir()]
    for domain_root in (path for path in dataset_root.iterdir() if path.is_dir()):
        subsets.extend(path for path in domain_root.iterdir() if (path / "images").is_dir())
    subsets = sorted(set(subsets))

    if args.subsets:
        selected_subsets = set(args.subsets)
        subsets = [path for path in subsets if path.name in selected_subsets]
        missing = selected_subsets - {path.name for path in subsets}
        if missing:
            raise FileNotFoundError(f"Subsets not found: {', '.join(sorted(missing))}")

    selected_chunks = set(args.chunks or [])
    shards: list[Shard] = []

    for subset_root in subsets:
        image_root = subset_root / "images"
        if not image_root.is_dir():
            raise FileNotFoundError(f"Missing images directory: {image_root}")

        for chunk_dir in sorted(image_root.glob("chunk-*")):
            if selected_chunks and chunk_dir.name not in selected_chunks:
                continue

            for camera_dir in sorted(chunk_dir.glob("observation.images.depth_*")):
                episodes = sorted(path for path in camera_dir.glob("episode_*") if path.is_dir())
                for start in range(0, len(episodes), args.episodes_per_shard):
                    shards.append(
                        Shard(
                            chunk=chunk_dir.name,
                            camera=camera_dir.name,
                            source_root=subset_root,
                            episodes=tuple(episodes[start : start + args.episodes_per_shard]),
                        )
                    )

    return shards


def output_path(shard: Shard) -> Path:
    first = shard.episodes[0].name.removeprefix("episode_")
    last = shard.episodes[-1].name.removeprefix("episode_")
    name = f"episodes-{first}-{last}.tar"
    return shard.source_root / "images" / shard.chunk / shard.camera / name


def episode_range(shard: Shard) -> str:
    first = shard.episodes[0].name.removeprefix("episode_")
    last = shard.episodes[-1].name.removeprefix("episode_")
    return f"{first}-{last}"


def format_duration(seconds: float) -> str:
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}h{minutes:02d}m{secs:02d}s"
    return f"{minutes:02d}m{secs:02d}s"


def delete_source_episodes(shard: Shard) -> int:
    image_root = (shard.source_root / "images").resolve()
    deleted = 0

    for episode in shard.episodes:
        if not episode.exists():
            continue
        if episode.is_symlink():
            raise RuntimeError(f"Refusing to delete symlink: {episode}")
        target = episode.resolve()
        if not target.is_relative_to(image_root) or not target.name.startswith("episode_"):
            raise RuntimeError(f"Refusing to delete unexpected path: {target}")
        shutil.rmtree(target)
        deleted += 1

    return deleted


def sync_file(path: Path) -> None:
    with path.open("rb") as file:
        os.fsync(file.fileno())


def create_shard(
    shard: Shard,
    overwrite: bool,
    delete_source: bool,
) -> tuple[str, Path, int, float]:
    started = time.monotonic()
    tar_path = output_path(shard)
    part_path = tar_path.with_suffix(".tar.part")
    tar_path.parent.mkdir(parents=True, exist_ok=True)

    if tar_path.exists() and not overwrite:
        deleted = delete_source_episodes(shard) if delete_source else 0
        return "skipped", tar_path, deleted, time.monotonic() - started

    tar_path.unlink(missing_ok=True)
    part_path.unlink(missing_ok=True)
    relative_episodes = [path.relative_to(shard.source_root).as_posix() for path in shard.episodes]

    list_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=tar_path.parent,
            prefix=f".{tar_path.stem}-",
            suffix=".txt",
            delete=False,
        ) as file:
            file.write("\n".join(relative_episodes) + "\n")
            list_path = Path(file.name)

        subprocess.run(
            [
                "tar",
                "--create",
                "--file",
                str(part_path),
                "--directory",
                str(shard.source_root),
                "--sort=name",
                "--verbatim-files-from",
                "--files-from",
                str(list_path),
            ],
            check=True,
        )
        sync_file(part_path)
        part_path.replace(tar_path)
        deleted = delete_source_episodes(shard) if delete_source else 0
        return "created", tar_path, deleted, time.monotonic() - started
    except Exception:
        part_path.unlink(missing_ok=True)
        raise
    finally:
        if list_path is not None:
            list_path.unlink(missing_ok=True)


def main() -> None:
    total_started = time.monotonic()
    args = parse_args()
    if args.episodes_per_shard < 1:
        raise ValueError("--episodes-per-shard must be at least 1")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if not args.dry_run and shutil.which("tar") is None:
        raise RuntimeError("GNU tar is required but was not found in PATH")

    shards = discover_shards(args)
    episode_camera_count = sum(len(shard.episodes) for shard in shards)
    dataset_root = args.dataset_root.resolve()

    print("Depth TAR packing", flush=True)
    print(f"  Dataset root:          {dataset_root}", flush=True)
    print(f"  TAR location:          in place", flush=True)
    print(f"  Delete source PNGs:    {'yes' if args.delete_source else 'no'}", flush=True)
    print(f"  Episodes per TAR:      {args.episodes_per_shard}", flush=True)
    print(f"  Workers:               {args.workers}", flush=True)
    print(f"  Planned TAR files:     {len(shards)}", flush=True)
    print(f"  Episode-camera dirs:   {episode_camera_count}", flush=True)
    print(flush=True)

    if args.dry_run:
        for shard in shards:
            print(output_path(shard))
        return

    created = 0
    skipped = 0
    deleted_total = 0
    total = len(shards)
    progress_width = max(1, len(str(total)))

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                create_shard,
                shard,
                args.overwrite,
                args.delete_source,
            ): shard
            for shard in shards
        }
        for position, future in enumerate(as_completed(futures), start=1):
            shard = futures[future]
            subset = shard.source_root.relative_to(dataset_root).as_posix()
            camera = shard.camera.removeprefix("observation.images.")
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            try:
                status, tar_path, deleted, elapsed = future.result()
            except Exception as error:
                for pending in futures:
                    pending.cancel()
                print(
                    f"[{timestamp}] [{position:0{progress_width}d}/{total}] FAILED  | "
                    f"{subset} | {shard.chunk} | {camera} | episodes {episode_range(shard)} | "
                    f"{error}",
                    flush=True,
                )
                raise

            created += status == "created"
            skipped += status == "skipped"
            deleted_total += deleted
            size_gib = tar_path.stat().st_size / 1024**3
            deletion = f" | deleted {deleted}" if args.delete_source else ""
            print(
                f"[{timestamp}] [{position:0{progress_width}d}/{total}] {status.upper():7} | "
                f"{subset} | {shard.chunk} | {camera} | episodes {episode_range(shard)} | "
                f"{size_gib:.2f} GiB{deletion} | {format_duration(elapsed)}",
                flush=True,
            )

    print(flush=True)
    print("Depth TAR packing complete", flush=True)
    print(f"  Created TAR files:     {created}", flush=True)
    print(f"  Skipped TAR files:     {skipped}", flush=True)
    if args.delete_source:
        print(f"  Deleted episode-camera dirs: {deleted_total}", flush=True)
    print(f"  Total elapsed:         {format_duration(time.monotonic() - total_started)}", flush=True)


if __name__ == "__main__":
    main()
