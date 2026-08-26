#!/usr/bin/env python3
"""Download selected subsets of Sponbebob4258/recam-lerobot."""

from __future__ import annotations

import argparse
import os
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_REPO_ID = "Sponbebob4258/recam-lerobot"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESTINATION = PROJECT_ROOT / "DATA" / "recam_lerobot"
DATASET_MARKERS = frozenset({"data", "images", "meta", "videos"})
MAX_SUBSET_DEPTH = 4


@dataclass(frozen=True)
class SubsetLayout:
    roots: tuple[str, ...]
    chunks_by_root: dict[str, tuple[int, ...]]


def normalize_subset(value: str) -> str:
    value = value.strip()
    if value == "all":
        return value
    if not value or value.startswith("/") or "\\" in value:
        raise ValueError(f"Invalid subset path: {value!r}")
    parts = value.split("/")
    if any(
        part in {"", ".", ".."} or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", part) is None
        for part in parts
    ):
        raise ValueError(f"Invalid subset path: {value!r}")
    return "/".join(parts)


def normalize_subsets(values: Iterable[str]) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()
    for raw_value in values:
        for item in raw_value.split(","):
            subset = normalize_subset(item)
            if subset not in seen:
                selected.append(subset)
                seen.add(subset)
    if "all" in seen and len(seen) != 1:
        raise ValueError("The 'all' selector cannot be combined with explicit subsets")
    return selected


def parse_chunks(value: str | None) -> list[int] | None:
    if value is None:
        return None
    chunks: set[int] = set()
    for raw_item in value.split(","):
        item = raw_item.strip()
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


def normalize_modality(value: str) -> str:
    value = value.strip()
    aliases = {"metadata": "meta", "parquet": "data"}
    value = aliases.get(value, value)
    if value in {"all", "data", "images", "meta", "videos"}:
        return value
    if value.startswith(("rgb_", "depth_", "normal_")):
        value = f"observation.images.{value}"
    if value.startswith("observation."):
        value = f"videos/{value}"
    parts = value.split("/")
    if len(parts) != 2 or parts[0] not in {"images", "videos"}:
        raise ValueError(f"Invalid modality selector: {value!r}")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", parts[1]) is None:
        raise ValueError(f"Invalid modality selector: {value!r}")
    return "/".join(parts)


def normalize_modalities(values: Iterable[str]) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()
    for raw_value in values:
        for item in raw_value.split(","):
            modality = normalize_modality(item)
            if modality not in seen:
                selected.append(modality)
                seen.add(modality)
    if "all" in seen:
        if len(seen) != 1:
            raise ValueError("The 'all' modality cannot be combined with explicit modalities")
        return []
    return selected


def is_directory_entry(entry: Any) -> bool:
    return (
        getattr(entry, "type", None) == "directory"
        or entry.__class__.__name__ == "RepoFolder"
    )


def list_directories(
    api: Any,
    repo_id: str,
    revision: str,
    path: str,
) -> list[str]:
    entries = api.list_repo_tree(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        path_in_repo=path or None,
        recursive=False,
    )
    return sorted(getattr(entry, "path") for entry in entries if is_directory_entry(entry))


def discover_subsets(api: Any, repo_id: str, revision: str) -> list[str]:
    queue = [""]
    visited: set[str] = set()
    subsets: set[str] = set()
    while queue:
        path = queue.pop(0)
        if path in visited:
            continue
        visited.add(path)
        directories = list_directories(api, repo_id, revision, path)
        direct_names = {directory.rsplit("/", 1)[-1] for directory in directories}
        if path and direct_names & DATASET_MARKERS:
            subsets.add(path)
            continue
        if path.count("/") + bool(path) >= MAX_SUBSET_DEPTH:
            continue
        queue.extend(directories)
    return sorted(subsets)


def discover_subset_layout(
    api: Any,
    repo_id: str,
    revision: str,
    subset: str,
) -> SubsetLayout:
    subset = normalize_subset(subset)
    root_paths = list_directories(api, repo_id, revision, subset)
    roots = tuple(path.rsplit("/", 1)[-1] for path in root_paths)
    chunks_by_root: dict[str, tuple[int, ...]] = {}
    for root in roots:
        child_paths = list_directories(api, repo_id, revision, f"{subset}/{root}")
        chunks = sorted(
            int(match.group(1))
            for path in child_paths
            if (match := re.fullmatch(r"chunk-(\d+)", path.rsplit("/", 1)[-1]))
        )
        if chunks:
            chunks_by_root[root] = tuple(chunks)
    return SubsetLayout(roots=roots, chunks_by_root=chunks_by_root)


def build_allow_patterns(
    subset: str,
    modalities: list[str],
    chunks: list[int] | None,
    layout: SubsetLayout | None,
) -> list[str]:
    subset = normalize_subset(subset)
    if not modalities and chunks is None:
        return [f"{subset}/**"]
    if layout is None:
        if not modalities:
            patterns = [f"{subset}/meta/**"]
            patterns.extend(f"{subset}/*/chunk-{chunk:03d}/**" for chunk in chunks or [])
            return patterns
        roots = {modality.split("/", 1)[0] for modality in modalities}
        chunked_roots = roots - {"meta"}
        chunks_by_root = {root: tuple(chunks or []) for root in chunked_roots}
    else:
        roots = set(layout.roots)
        chunks_by_root = layout.chunks_by_root

    patterns: list[str] = []
    if not modalities:
        assert chunks is not None
        found_chunks: set[int] = set()
        for root in sorted(roots):
            available = set(chunks_by_root.get(root, ()))
            if not available:
                patterns.append(f"{subset}/{root}/**")
                continue
            for chunk in chunks:
                if chunk in available:
                    patterns.append(f"{subset}/{root}/chunk-{chunk:03d}/**")
                    found_chunks.add(chunk)
        missing = sorted(set(chunks) - found_chunks)
        if missing:
            raise ValueError(
                f"Chunk(s) not found in {subset}: "
                f"{', '.join(f'{chunk:03d}' for chunk in missing)}"
            )
        return patterns

    for modality in modalities:
        root, separator, nested = modality.partition("/")
        if layout is not None and root not in roots:
            raise ValueError(f"Modality root {root!r} does not exist in subset {subset}")
        available = set(chunks_by_root.get(root, ()))
        if nested and not available and layout is not None:
            raise ValueError(f"Modality {modality!r} is not chunked in subset {subset}")
        if not available:
            patterns.append(f"{subset}/{root}/**")
            continue
        if chunks is None:
            if separator:
                patterns.append(f"{subset}/{root}/chunk-*/{nested}/**")
            else:
                patterns.append(f"{subset}/{root}/**")
            continue
        missing = sorted(set(chunks) - available)
        if missing:
            raise ValueError(
                f"Chunk(s) unavailable for {subset}/{root}: "
                f"{', '.join(f'{chunk:03d}' for chunk in missing)}"
            )
        for chunk in chunks:
            suffix = f"/{nested}" if separator else ""
            patterns.append(f"{subset}/{root}/chunk-{chunk:03d}{suffix}/**")
    return list(dict.fromkeys(patterns))


def validate_remote_nested_modalities(
    api: Any,
    repo_id: str,
    revision: str,
    subset: str,
    modalities: list[str],
    chunks: list[int] | None,
    layout: SubsetLayout,
) -> None:
    cache: dict[str, set[str]] = {}
    for modality in modalities:
        root, separator, nested = modality.partition("/")
        if not separator:
            continue
        available_chunks = layout.chunks_by_root.get(root, ())
        chunks_to_check = chunks or list(available_chunks[:1])
        for chunk in chunks_to_check:
            path = f"{subset}/{root}/chunk-{chunk:03d}"
            if path not in cache:
                cache[path] = {
                    directory.rsplit("/", 1)[-1]
                    for directory in list_directories(api, repo_id, revision, path)
                }
            if nested not in cache[path]:
                raise ValueError(
                    f"Modality {nested!r} does not exist in {path}; "
                    f"available: {', '.join(sorted(cache[path])) or '<none>'}"
                )


def validate_local_patterns(destination: Path, patterns: list[str]) -> None:
    missing: list[str] = []
    for pattern in patterns:
        base_pattern = pattern.removesuffix("/**")
        found = False
        for base in destination.glob(base_pattern):
            if base.is_file() or (base.is_dir() and any(path.is_file() for path in base.rglob("*"))):
                found = True
                break
        if not found:
            missing.append(pattern)
    if missing:
        raise RuntimeError(f"Downloaded files do not match: {', '.join(missing)}")


def download_subset(
    snapshot_download: Callable[..., str],
    *,
    repo_id: str,
    revision: str,
    destination: Path,
    subset: str,
    patterns: list[str],
    workers: int,
    max_attempts: int,
    retry_delay_seconds: float,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    for attempt in range(1, max_attempts + 1):
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                revision=revision,
                local_dir=destination,
                allow_patterns=patterns,
                max_workers=workers,
            )
            validate_local_patterns(destination, patterns)
            return
        except Exception as exc:
            if attempt == max_attempts:
                raise RuntimeError(
                    f"Subset {subset} failed after {max_attempts} attempt(s): {exc}"
                ) from exc
            delay = retry_delay_seconds * attempt
            print(
                f"RETRY {attempt + 1}/{max_attempts}: {subset}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            if delay:
                sleep(delay)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download one or more recam-lerobot subsets while preserving the repository "
            "layout below ./DATA/recam_lerobot."
        )
    )
    parser.add_argument(
        "subsets",
        nargs="*",
        help=(
            "Subset paths, comma-separated or space-separated; for example "
            "real_world/droid simulation/libero. Use 'all' for every discovered subset."
        ),
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument(
        "--modalities",
        action="append",
        default=[],
        metavar="LIST",
        help=(
            "Comma-separated content filters: data, meta, videos, images, or an exact "
            "video stream such as rgb_01 or observation.images.rgb_01"
        ),
    )
    parser.add_argument(
        "--chunks",
        help="Chunk selector such as 0,2,5-7 or chunk-000",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("RECAM_DOWNLOAD_WORKERS", "4")),
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=int(os.environ.get("RECAM_DOWNLOAD_MAX_ATTEMPTS", "3")),
    )
    parser.add_argument(
        "--retry-delay-seconds",
        type=float,
        default=float(os.environ.get("RECAM_DOWNLOAD_RETRY_DELAY_SECONDS", "5")),
    )
    parser.add_argument(
        "--list-subsets",
        action="store_true",
        help="List remotely discovered subsets without downloading files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print local destination and allow-patterns without network access",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.max_attempts < 1:
        raise ValueError("--max-attempts must be positive")
    if args.retry_delay_seconds < 0:
        raise ValueError("--retry-delay-seconds cannot be negative")
    selected = normalize_subsets(args.subsets)
    modalities = normalize_modalities(args.modalities)
    chunks = parse_chunks(args.chunks)
    if args.list_subsets and (selected or modalities or chunks is not None):
        raise ValueError(
            "--list-subsets cannot be combined with subset, modality, or chunk arguments"
        )
    if not args.list_subsets and not selected:
        raise ValueError("Select at least one subset, use 'all', or pass --list-subsets")

    destination = args.destination.expanduser().resolve()
    print(f"Repository:  {args.repo_id}")
    print(f"Revision:    {args.revision}")
    print(f"Destination: {destination}")
    print(f"Workers:     {args.workers}")
    print(f"Modalities:  {', '.join(modalities) if modalities else '<all>'}")
    print(
        "Chunks:      "
        + (", ".join(f"{chunk:03d}" for chunk in chunks) if chunks is not None else "<all>")
    )
    if args.dry_run:
        if args.list_subsets:
            print("Dry run: remote subset discovery was not requested.")
        elif selected == ["all"]:
            print("  <all remotely discovered subsets>")
        else:
            for subset in selected:
                for pattern in build_allow_patterns(subset, modalities, chunks, layout=None):
                    print(f"  {pattern}")
        return

    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    resolved_revision = api.dataset_info(args.repo_id, revision=args.revision).sha
    if not resolved_revision:
        raise RuntimeError(f"Could not resolve revision {args.revision!r}")
    available = discover_subsets(api, args.repo_id, resolved_revision)
    if args.list_subsets:
        print(f"Resolved:    {resolved_revision}")
        for subset in available:
            print(subset)
        return
    if selected == ["all"]:
        selected = available
    missing = sorted(set(selected) - set(available))
    if missing:
        available_text = ", ".join(available) or "<none discovered>"
        raise ValueError(
            f"Unknown subset(s): {', '.join(missing)}. Available subsets: {available_text}"
        )

    destination.mkdir(parents=True, exist_ok=True)
    print(f"Resolved:    {resolved_revision}")
    print(f"Subsets:     {', '.join(selected)}")
    plans: dict[str, list[str]] = {}
    for subset in selected:
        layout = (
            discover_subset_layout(api, args.repo_id, resolved_revision, subset)
            if modalities or chunks is not None
            else None
        )
        plans[subset] = build_allow_patterns(subset, modalities, chunks, layout)
        if layout is not None:
            validate_remote_nested_modalities(
                api,
                args.repo_id,
                resolved_revision,
                subset,
                modalities,
                chunks,
                layout,
            )
        for pattern in plans[subset]:
            print(f"  {pattern}")
    for index, subset in enumerate(selected, start=1):
        print(f"[{index}/{len(selected)}] Downloading {subset}", flush=True)
        download_subset(
            snapshot_download,
            repo_id=args.repo_id,
            revision=resolved_revision,
            destination=destination,
            subset=subset,
            patterns=plans[subset],
            workers=args.workers,
            max_attempts=args.max_attempts,
            retry_delay_seconds=args.retry_delay_seconds,
        )
        print(f"[{index}/{len(selected)}] COMPLETE {subset}", flush=True)
    print("Dataset download complete.", flush=True)


if __name__ == "__main__":
    main()
