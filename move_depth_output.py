#!/usr/bin/env python3
"""Move completed depth images and metadata into a LeRobot dataset."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, order=True)
class DepthItem:
    chunk: str
    camera: str
    episode: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Move depth images and FoundationStereo metadata into a dataset."
    )
    parser.add_argument("source_output", type=Path, help="Depth conversion output directory")
    parser.add_argument("target_dataset", type=Path, help="Target LeRobot dataset root")
    parser.add_argument(
        "--chunks",
        nargs="+",
        help="Chunk names to move, for example chunk-000 chunk-001; default: all",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and show the move plan without changing files",
    )
    return parser.parse_args()


def discover_items(source: Path, selected_chunks: set[str]) -> list[DepthItem]:
    items: set[DepthItem] = set()

    image_root = source / "images"
    for path in image_root.glob("chunk-*/observation.images.depth_*/episode_*"):
        if path.is_dir() and (not selected_chunks or path.parents[1].name in selected_chunks):
            items.add(DepthItem(path.parents[1].name, path.parent.name, path.name))

    metadata_root = source / "annotations/foundation_stereo_depth"
    for path in metadata_root.glob("chunk-*/observation.images.depth_*/episode_*.json"):
        if path.is_file() and (not selected_chunks or path.parents[1].name in selected_chunks):
            items.add(DepthItem(path.parents[1].name, path.parent.name, path.stem))

    return sorted(items)


def paths_for(item: DepthItem, source: Path, target: Path) -> tuple[tuple[Path, Path], ...]:
    image_relative = Path("images") / item.chunk / item.camera / item.episode
    metadata_relative = (
        Path("annotations/foundation_stereo_depth")
        / item.chunk
        / item.camera
        / f"{item.episode}.json"
    )
    return (
        (source / image_relative, target / image_relative),
        (source / metadata_relative, target / metadata_relative),
    )


def validate_plan(items: list[DepthItem], source: Path, target: Path) -> None:
    errors = []
    for item in items:
        for source_path, target_path in paths_for(item, source, target):
            source_exists = source_path.exists()
            target_exists = target_path.exists()
            if source_exists and target_exists:
                errors.append(f"Source and target both exist: {source_path} -> {target_path}")
            elif not source_exists and not target_exists:
                errors.append(f"Missing from source and target: {source_path}")

    if errors:
        preview = "\n".join(errors[:20])
        remainder = len(errors) - 20
        suffix = f"\n... and {remainder} more" if remainder > 0 else ""
        raise RuntimeError(f"Move plan is not safe:\n{preview}{suffix}")


def move_path(source: Path, target: Path) -> bool:
    if not source.exists():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, target)
    return True


def remove_empty_directories(root: Path) -> None:
    if not root.is_dir():
        return
    directories = sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for path in directories:
        try:
            path.rmdir()
        except OSError:
            pass
    try:
        root.rmdir()
    except OSError:
        pass


def main() -> None:
    args = parse_args()
    source = args.source_output.resolve()
    target = args.target_dataset.resolve()

    if not source.is_dir():
        raise FileNotFoundError(f"Source output does not exist: {source}")
    if not target.is_dir():
        raise FileNotFoundError(f"Target dataset does not exist: {target}")
    if source == target or source in target.parents or target in source.parents:
        raise RuntimeError("Source and target must be separate directories")
    if source.stat().st_dev != target.stat().st_dev:
        raise RuntimeError("Source and target must be on the same filesystem")

    selected_chunks = set(args.chunks or [])
    items = discover_items(source, selected_chunks)
    if not items:
        raise RuntimeError("No depth images or metadata were found")
    validate_plan(items, source, target)

    print("Depth output move")
    print(f"  Source:              {source}")
    print(f"  Target:              {target}")
    print(f"  Depth episode items: {len(items)}")
    print(f"  Chunks:              {', '.join(sorted({item.chunk for item in items}))}")
    print("  Logs:                ignored")

    if args.dry_run:
        print("  Mode:                dry run")
        print("Validation complete; no files were moved.")
        return

    moved_images = 0
    moved_metadata = 0
    total = len(items)
    width = len(str(total))

    for index, item in enumerate(items, start=1):
        image_pair, metadata_pair = paths_for(item, source, target)
        moved_images += move_path(*image_pair)
        moved_metadata += move_path(*metadata_pair)

        if index == 1 or index % 100 == 0 or index == total:
            print(
                f"[{index:0{width}d}/{total}] {item.chunk} | {item.camera} | {item.episode}",
                flush=True,
            )

    remove_empty_directories(source / "images")
    remove_empty_directories(source / "annotations/foundation_stereo_depth")

    print("Depth output move complete")
    print(f"  Moved image directories: {moved_images}")
    print(f"  Moved metadata files:    {moved_metadata}")
    print(f"  Source logs untouched:   {(source / 'logs').is_dir()}")


if __name__ == "__main__":
    main()
