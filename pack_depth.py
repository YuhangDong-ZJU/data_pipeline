#!/usr/bin/env python3
"""Pack LeRobot depth PNG directories into episode-aligned TAR shards."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Shard:
    dataset_path: Path
    chunk: str
    camera: str
    source_root: Path
    episodes: tuple[Path, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pack depth episode directories into uncompressed TAR shards."
    )
    parser.add_argument("input_root", type=Path, help="Root containing LeRobot subsets")
    parser.add_argument("output_root", type=Path, help="Directory for TAR shards")
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
    input_root = args.input_root.resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    subsets = [path for path in input_root.iterdir() if (path / "images").is_dir()]
    for domain_root in (path for path in input_root.iterdir() if path.is_dir()):
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
                            dataset_path=subset_root.relative_to(input_root),
                            chunk=chunk_dir.name,
                            camera=camera_dir.name,
                            source_root=subset_root,
                            episodes=tuple(episodes[start : start + args.episodes_per_shard]),
                        )
                    )

    return shards


def output_path(shard: Shard, output_root: Path) -> Path:
    first = shard.episodes[0].name.removeprefix("episode_")
    last = shard.episodes[-1].name.removeprefix("episode_")
    name = f"episodes-{first}-{last}.tar"
    return output_root / shard.dataset_path / shard.chunk / shard.camera / name


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
    output_root: Path,
    overwrite: bool,
    delete_source: bool,
) -> tuple[str, Path, int]:
    tar_path = output_path(shard, output_root)
    part_path = tar_path.with_suffix(".tar.part")
    tar_path.parent.mkdir(parents=True, exist_ok=True)

    if tar_path.exists() and not overwrite:
        deleted = delete_source_episodes(shard) if delete_source else 0
        return "skipped", tar_path, deleted

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
        return "created", tar_path, deleted
    except Exception:
        part_path.unlink(missing_ok=True)
        raise
    finally:
        if list_path is not None:
            list_path.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    if args.episodes_per_shard < 1:
        raise ValueError("--episodes-per-shard must be at least 1")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if not args.dry_run and shutil.which("tar") is None:
        raise RuntimeError("GNU tar is required but was not found in PATH")

    shards = discover_shards(args)
    episode_camera_count = sum(len(shard.episodes) for shard in shards)
    output_root = args.output_root.resolve()

    print(f"Planned TAR files: {len(shards)}")
    print(f"Episode-camera directories: {episode_camera_count}")

    if args.dry_run:
        for shard in shards:
            print(output_path(shard, output_root))
        return

    output_root.mkdir(parents=True, exist_ok=True)
    created = 0
    skipped = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                create_shard,
                shard,
                output_root,
                args.overwrite,
                args.delete_source,
            ): shard
            for shard in shards
        }
        for position, future in enumerate(as_completed(futures), start=1):
            status, tar_path, deleted = future.result()
            created += status == "created"
            skipped += status == "skipped"
            suffix = f", deleted={deleted}" if args.delete_source else ""
            print(f"[{position}/{len(shards)}] {status}{suffix}: {tar_path}", flush=True)

    print(f"Complete. created={created}, skipped={skipped}")


if __name__ == "__main__":
    main()
