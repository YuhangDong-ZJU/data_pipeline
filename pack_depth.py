#!/usr/bin/env python3
"""Pack LeRobot depth PNG directories into episode-aligned TAR shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Shard:
    subset: str
    chunk: str
    camera: str
    index: int
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
        help="Recreate existing complete shards",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the planned shards without writing files",
    )
    return parser.parse_args()


def discover_shards(args: argparse.Namespace) -> list[Shard]:
    input_root = args.input_root.resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    if args.subsets:
        subsets = [input_root / name for name in args.subsets]
    else:
        subsets = sorted(path for path in input_root.iterdir() if (path / "images").is_dir())

    selected_chunks = set(args.chunks or [])
    shards: list[Shard] = []

    for subset_root in subsets:
        image_root = subset_root / "images"
        if not image_root.is_dir():
            raise FileNotFoundError(f"Missing images directory: {image_root}")

        for chunk_dir in sorted(image_root.glob("chunk-*")):
            if selected_chunks and chunk_dir.name not in selected_chunks:
                continue

            cameras = sorted(chunk_dir.glob("observation.images.depth_*"))
            for camera_dir in cameras:
                episodes = sorted(path for path in camera_dir.glob("episode_*") if path.is_dir())
                for start in range(0, len(episodes), args.episodes_per_shard):
                    group = tuple(episodes[start : start + args.episodes_per_shard])
                    shards.append(
                        Shard(
                            subset=subset_root.name,
                            chunk=chunk_dir.name,
                            camera=camera_dir.name,
                            index=start // args.episodes_per_shard,
                            source_root=subset_root,
                            episodes=group,
                        )
                    )

    return shards


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def paths_for(shard: Shard, output_root: Path) -> tuple[Path, Path, Path]:
    directory = output_root / shard.subset / shard.chunk / shard.camera
    stem = f"shard-{shard.index:04d}"
    return directory / f"{stem}.tar", directory / f"{stem}.json", directory / f"{stem}.tar.part"


def expected_episode_names(shard: Shard) -> list[str]:
    return [path.name for path in shard.episodes]


def existing_shard_is_valid(shard: Shard, tar_path: Path, metadata_path: Path) -> bool:
    if not tar_path.is_file() or not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        metadata.get("subset") == shard.subset
        and metadata.get("chunk") == shard.chunk
        and metadata.get("camera") == shard.camera
        and metadata.get("episodes") == expected_episode_names(shard)
        and metadata.get("archive_size_bytes") == tar_path.stat().st_size
    )


def create_shard(shard: Shard, output_root: Path, overwrite: bool) -> tuple[str, Path]:
    tar_path, metadata_path, part_path = paths_for(shard, output_root)
    tar_path.parent.mkdir(parents=True, exist_ok=True)

    if not overwrite and existing_shard_is_valid(shard, tar_path, metadata_path):
        return "skipped", metadata_path

    if not overwrite and (tar_path.exists() or metadata_path.exists()):
        raise RuntimeError(
            f"Incomplete or mismatched shard exists: {tar_path}. "
            "Inspect it or rerun with --overwrite."
        )

    tar_path.unlink(missing_ok=True)
    metadata_path.unlink(missing_ok=True)
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
            file.write("\n".join(relative_episodes))
            file.write("\n")
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
        part_path.replace(tar_path)

        metadata = {
            "format_version": 1,
            "archive": tar_path.name,
            "archive_size_bytes": tar_path.stat().st_size,
            "sha256": sha256(tar_path),
            "subset": shard.subset,
            "chunk": shard.chunk,
            "camera": shard.camera,
            "episode_count": len(shard.episodes),
            "episodes": expected_episode_names(shard),
        }
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return "created", metadata_path
    except Exception:
        part_path.unlink(missing_ok=True)
        raise
    finally:
        if list_path is not None:
            list_path.unlink(missing_ok=True)


def write_manifest(output_root: Path) -> Path:
    records = []
    for path in sorted(output_root.glob("*/*/*/shard-*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        record["archive_path"] = (path.parent / record["archive"]).relative_to(output_root).as_posix()
        records.append(record)

    manifest_path = output_root / "manifest.jsonl"
    temporary_path = output_root / "manifest.jsonl.part"
    with temporary_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary_path.replace(manifest_path)
    return manifest_path


def main() -> None:
    args = parse_args()
    if args.episodes_per_shard < 1:
        raise ValueError("--episodes-per-shard must be at least 1")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if shutil.which("tar") is None:
        raise RuntimeError("GNU tar is required but was not found in PATH")

    shards = discover_shards(args)
    episode_count = sum(len(shard.episodes) for shard in shards)
    print(f"Planned shards: {len(shards)}")
    print(f"Episode-camera directories: {episode_count}")
    if args.dry_run:
        for shard in shards:
            first = shard.episodes[0].name
            last = shard.episodes[-1].name
            print(
                f"{shard.subset}/{shard.chunk}/{shard.camera}/shard-{shard.index:04d}.tar "
                f"({first}..{last})"
            )
        return

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    created = 0
    skipped = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(create_shard, shard, output_root, args.overwrite): shard
            for shard in shards
        }
        for position, future in enumerate(as_completed(futures), start=1):
            status, metadata_path = future.result()
            created += status == "created"
            skipped += status == "skipped"
            print(f"[{position}/{len(shards)}] {status}: {metadata_path}", flush=True)

    manifest_path = write_manifest(output_root)
    print(f"Complete. created={created}, skipped={skipped}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
