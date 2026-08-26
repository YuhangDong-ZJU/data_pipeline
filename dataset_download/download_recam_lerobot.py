#!/usr/bin/env python3
"""Download selected subsets of Sponbebob4258/recam-lerobot."""

from __future__ import annotations

import argparse
import os
import re
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any


DEFAULT_REPO_ID = "Sponbebob4258/recam-lerobot"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESTINATION = PROJECT_ROOT / "DATA" / "recam_lerobot"
DATASET_MARKERS = frozenset({"data", "images", "meta", "videos"})
MAX_SUBSET_DEPTH = 4


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


def is_directory_entry(entry: Any) -> bool:
    return (
        getattr(entry, "type", None) == "directory"
        or entry.__class__.__name__ == "RepoFolder"
    )


def discover_subsets(api: Any, repo_id: str, revision: str) -> list[str]:
    queue = [""]
    visited: set[str] = set()
    subsets: set[str] = set()
    while queue:
        path = queue.pop(0)
        if path in visited:
            continue
        visited.add(path)
        entries = api.list_repo_tree(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            path_in_repo=path or None,
            recursive=False,
        )
        directories = sorted(
            getattr(entry, "path") for entry in entries if is_directory_entry(entry)
        )
        direct_names = {directory.rsplit("/", 1)[-1] for directory in directories}
        if path and direct_names & DATASET_MARKERS:
            subsets.add(path)
            continue
        if path.count("/") + bool(path) >= MAX_SUBSET_DEPTH:
            continue
        queue.extend(directories)
    return sorted(subsets)


def allow_patterns(subset: str) -> list[str]:
    return [f"{normalize_subset(subset)}/**"]


def validate_local_subset(destination: Path, subset: str) -> None:
    subset_root = destination.joinpath(*subset.split("/"))
    if not subset_root.is_dir():
        raise RuntimeError(f"Downloaded subset directory is missing: {subset_root}")
    if not any(path.is_file() for path in subset_root.rglob("*")):
        raise RuntimeError(f"Downloaded subset contains no files: {subset_root}")


def download_subset(
    snapshot_download: Callable[..., str],
    *,
    repo_id: str,
    revision: str,
    destination: Path,
    subset: str,
    workers: int,
    max_attempts: int,
    retry_delay_seconds: float,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    patterns = allow_patterns(subset)
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
            validate_local_subset(destination, subset)
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
    if args.list_subsets and selected:
        raise ValueError("--list-subsets cannot be combined with subset arguments")
    if not args.list_subsets and not selected:
        raise ValueError("Select at least one subset, use 'all', or pass --list-subsets")

    destination = args.destination.expanduser().resolve()
    print(f"Repository:  {args.repo_id}")
    print(f"Revision:    {args.revision}")
    print(f"Destination: {destination}")
    print(f"Workers:     {args.workers}")
    if args.dry_run:
        if args.list_subsets:
            print("Dry run: remote subset discovery was not requested.")
        elif selected == ["all"]:
            print("  <all remotely discovered subsets>")
        else:
            for subset in selected:
                print(f"  {allow_patterns(subset)[0]}")
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
    for index, subset in enumerate(selected, start=1):
        print(f"[{index}/{len(selected)}] Downloading {subset}", flush=True)
        download_subset(
            snapshot_download,
            repo_id=args.repo_id,
            revision=resolved_revision,
            destination=destination,
            subset=subset,
            workers=args.workers,
            max_attempts=args.max_attempts,
            retry_delay_seconds=args.retry_delay_seconds,
        )
        print(f"[{index}/{len(selected)}] COMPLETE {subset}", flush=True)
    print("Dataset download complete.", flush=True)


if __name__ == "__main__":
    main()
