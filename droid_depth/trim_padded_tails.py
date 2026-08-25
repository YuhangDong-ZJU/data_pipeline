#!/usr/bin/env python3
"""Remove padded tail timesteps from every modality of affected episodes."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


@dataclass
class TrimSpec:
    episode_index: int
    original_length: int
    trim_count: int
    padded_cameras: set[str] = field(default_factory=set)

    @property
    def final_length(self) -> int:
        return self.original_length - self.trim_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trim padded tail timesteps from a LeRobot v2.1 dataset."
    )
    parser.add_argument("dataset_root", type=Path, help="LeRobot v2.1 dataset root")
    parser.add_argument(
        "metadata_roots",
        type=Path,
        nargs="+",
        help="FoundationStereo metadata directories or conversion output roots",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and show the trim plan without changing files",
    )
    return parser.parse_args()


def metadata_root(path: Path) -> Path:
    path = path.resolve()
    nested = path / "annotations/foundation_stereo_depth"
    if nested.is_dir():
        return nested
    if path.is_dir() and path.name == "foundation_stereo_depth":
        return path
    raise FileNotFoundError(f"FoundationStereo metadata not found: {path}")


def load_trim_specs(roots: list[Path]) -> dict[int, TrimSpec]:
    specs: dict[int, TrimSpec] = {}

    for root in roots:
        for path in sorted(root.glob("chunk-*/observation.images.depth_*/episode_*.json")):
            value = json.loads(path.read_text(encoding="utf-8"))
            source = value["source"]
            missing = int(source["tail_missing_count"])
            if missing == 0:
                continue

            episode_index = int(source["episode_index"])
            frame_count = int(source["frame_count"])
            missing_indices = [int(index) for index in source["missing_frame_indices"]]
            expected_indices = list(range(frame_count - missing, frame_count))
            if missing_indices != expected_indices:
                raise RuntimeError(f"Missing frames are not a pure tail in {path}")

            camera = path.parent.name
            existing = specs.get(episode_index)
            if existing is None:
                specs[episode_index] = TrimSpec(
                    episode_index=episode_index,
                    original_length=frame_count,
                    trim_count=missing,
                    padded_cameras={camera},
                )
            else:
                if existing.original_length != frame_count:
                    raise RuntimeError(f"Conflicting frame counts for episode {episode_index}")
                existing.trim_count = max(existing.trim_count, missing)
                existing.padded_cameras.add(camera)

    if not specs:
        raise RuntimeError("No padded tail episodes were found")
    return specs


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tail-trim.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tail-trim.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def video_frame_count(path: Path, ffprobe: str) -> int:
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_frames",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    output = subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()
    if output.isdigit():
        return int(output)

    command[5:5] = ["-count_frames"]
    command[command.index("stream=nb_frames")] = "stream=nb_read_frames"
    output = subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()
    if not output.isdigit():
        raise RuntimeError(f"Cannot determine video frame count: {path}")
    return int(output)


def depth_frame_paths(directory: Path) -> list[Path]:
    frames = sorted(directory.glob("frame_*.png"))
    expected = [f"frame_{index:06d}.png" for index in range(len(frames))]
    if [path.name for path in frames] != expected:
        raise RuntimeError(f"Depth frame names are not contiguous: {directory}")
    return frames


def parquet_path(dataset: Path, info: dict[str, Any], episode_index: int) -> Path:
    chunk = episode_index // int(info["chunks_size"])
    relative = info["data_path"].format(
        episode_chunk=chunk,
        episode_index=episode_index,
    )
    return dataset / relative


def video_path(
    dataset: Path,
    info: dict[str, Any],
    episode_index: int,
    video_key: str,
) -> Path:
    chunk = episode_index // int(info["chunks_size"])
    relative = info["video_path"].format(
        episode_chunk=chunk,
        episode_index=episode_index,
        video_key=video_key,
    )
    return dataset / relative


def depth_directories(dataset: Path, info: dict[str, Any], episode_index: int) -> list[Path]:
    chunk = episode_index // int(info["chunks_size"])
    chunk_dir = dataset / "images" / f"chunk-{chunk:03d}"
    return sorted(
        path / f"episode_{episode_index:06d}"
        for path in chunk_dir.glob("observation.images.depth_*")
        if (path / f"episode_{episode_index:06d}").is_dir()
    )


def validate_media(
    dataset: Path,
    info: dict[str, Any],
    specs: dict[int, TrimSpec],
    episode_rows: dict[int, dict[str, Any]],
    video_keys: list[str],
    ffprobe: str,
) -> tuple[list[tuple[Path, int]], list[Path], int]:
    videos_to_trim: list[tuple[Path, int]] = []
    depth_frames_to_delete: list[Path] = []
    already_trimmed = 0

    for episode_index, spec in sorted(specs.items()):
        episode_row = episode_rows.get(episode_index)
        if episode_row is None:
            raise RuntimeError(f"Episode missing from meta/episodes.jsonl: {episode_index}")
        current_length = int(episode_row["length"])
        if current_length == spec.final_length:
            already_trimmed += 1
        elif current_length != spec.original_length:
            raise RuntimeError(
                f"Unexpected metadata length for episode {episode_index}: {current_length}"
            )

        table_path = parquet_path(dataset, info, episode_index)
        if not table_path.is_file():
            raise FileNotFoundError(table_path)
        table_rows = pq.ParquetFile(table_path).metadata.num_rows
        if table_rows not in (spec.original_length, spec.final_length):
            raise RuntimeError(f"Unexpected Parquet length for episode {episode_index}: {table_rows}")

        for key in video_keys:
            path = video_path(dataset, info, episode_index, key)
            if not path.is_file():
                raise FileNotFoundError(path)
            frames = video_frame_count(path, ffprobe)
            if frames == spec.original_length:
                videos_to_trim.append((path, spec.final_length))
            elif frames != spec.final_length:
                raise RuntimeError(f"Unexpected video length {frames}: {path}")

        directories = depth_directories(dataset, info, episode_index)
        if not directories:
            raise RuntimeError(
                f"No depth images found for episode {episode_index}; move depth outputs first"
            )
        existing_keys = {path.parent.name for path in directories}
        if not spec.padded_cameras.issubset(existing_keys):
            raise RuntimeError(f"Padded depth camera missing for episode {episode_index}")

        for directory in directories:
            frames = depth_frame_paths(directory)
            if len(frames) == spec.original_length:
                depth_frames_to_delete.extend(frames[spec.final_length :])
            elif len(frames) != spec.final_length:
                raise RuntimeError(f"Unexpected depth length {len(frames)}: {directory}")

    return videos_to_trim, depth_frames_to_delete, already_trimmed


def trim_video(path: Path, final_length: int, ffmpeg: str, ffprobe: str) -> None:
    temporary = path.with_name(f".{path.stem}.tail-trim.tmp.mp4")
    temporary.unlink(missing_ok=True)
    try:
        subprocess.run(
            [
                ffmpeg,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(path),
                "-map",
                "0:v:0",
                "-frames:v",
                str(final_length),
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                str(temporary),
            ],
            check=True,
        )
        if video_frame_count(temporary, ffprobe) != final_length:
            raise RuntimeError(f"Trimmed video validation failed: {path}")
        with temporary.open("rb") as file:
            os.fsync(file.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def column_values(column: pa.ChunkedArray) -> np.ndarray:
    array: pa.Array = column.combine_chunks()
    shape = []
    value_type = array.type
    while pa.types.is_fixed_size_list(value_type):
        shape.append(value_type.list_size)
        array = array.values
        value_type = value_type.value_type
    if array.null_count:
        raise RuntimeError("Null values are not supported in dataset statistics")
    values = np.asarray(array.to_numpy(zero_copy_only=False), dtype=np.float64)
    if shape:
        return values.reshape((len(column), *shape))
    return values.reshape((len(column), 1))


def update_stats(
    table: pa.Table,
    global_accumulator: dict[str, dict[str, Any]],
    feature_names: set[str] | None = None,
) -> dict[str, Any]:
    result = {}
    for name in table.column_names:
        if feature_names is not None and name not in feature_names:
            continue
        values = column_values(table[name])
        count = values.shape[0]
        minimum = values.min(axis=0)
        maximum = values.max(axis=0)
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        result[name] = {
            "min": minimum.tolist(),
            "max": maximum.tolist(),
            "mean": mean.tolist(),
            "std": std.tolist(),
            "count": [count],
        }

        accumulator = global_accumulator.get(name)
        if accumulator is None:
            accumulator = {
                "min": minimum.copy(),
                "max": maximum.copy(),
                "sum": values.sum(axis=0),
                "sum_sq": np.square(values).sum(axis=0),
                "count": count,
            }
            global_accumulator[name] = accumulator
        else:
            accumulator["min"] = np.minimum(accumulator["min"], minimum)
            accumulator["max"] = np.maximum(accumulator["max"], maximum)
            accumulator["sum"] += values.sum(axis=0)
            accumulator["sum_sq"] += np.square(values).sum(axis=0)
            accumulator["count"] += count
    return result


def finalize_global_stats(accumulators: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for name, value in accumulators.items():
        count = int(value["count"])
        mean = value["sum"] / count
        variance = np.maximum(value["sum_sq"] / count - np.square(mean), 0.0)
        result[name] = {
            "min": value["min"].tolist(),
            "max": value["max"].tolist(),
            "mean": mean.tolist(),
            "std": np.sqrt(variance).tolist(),
            "count": [count],
        }
    return result


def write_parquet_atomic(path: Path, table: pa.Table, compression: str) -> None:
    temporary = path.with_name(f".{path.name}.tail-trim.tmp")
    temporary.unlink(missing_ok=True)
    try:
        pq.write_table(table, temporary, compression=compression.lower())
        with temporary.open("rb") as file:
            os.fsync(file.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def rebuild_parquet_and_metadata(
    dataset: Path,
    info: dict[str, Any],
    episode_rows: list[dict[str, Any]],
    specs: dict[int, TrimSpec],
) -> tuple[int, int]:
    global_index = 0
    rewritten = 0
    accumulators: dict[str, dict[str, Any]] = {}
    stats_path = dataset / "meta/episodes_stats.jsonl"
    stats_temporary = stats_path.with_name(f".{stats_path.name}.tail-trim.tmp")

    with stats_path.open("r", encoding="utf-8") as file:
        first_stats_row = json.loads(next(line for line in file if line.strip()))
    stats_feature_names = set(first_stats_row["stats"])

    try:
        with stats_temporary.open("w", encoding="utf-8") as stats_file:
            total = len(episode_rows)
            width = len(str(total))
            for position, episode_row in enumerate(episode_rows, start=1):
                episode_index = int(episode_row["episode_index"])
                path = parquet_path(dataset, info, episode_index)
                parquet_file = pq.ParquetFile(path)
                compression = parquet_file.metadata.row_group(0).column(0).compression
                table = parquet_file.read()
                original_rows = table.num_rows
                spec = specs.get(episode_index)

                if spec:
                    if original_rows == spec.original_length:
                        table = table.slice(0, spec.final_length)
                    elif original_rows != spec.final_length:
                        raise RuntimeError(
                            f"Unexpected Parquet rows for episode {episode_index}: {original_rows}"
                        )
                elif original_rows != int(episode_row["length"]):
                    raise RuntimeError(
                        f"Parquet/meta length mismatch for episode {episode_index}"
                    )

                rows = table.num_rows
                frame_index = np.asarray(
                    table["frame_index"].combine_chunks().to_numpy(zero_copy_only=False)
                )
                if not np.array_equal(frame_index, np.arange(rows)):
                    raise RuntimeError(f"Invalid frame_index in episode {episode_index}")

                expected_index = np.arange(global_index, global_index + rows, dtype=np.int64)
                current_index = np.asarray(
                    table["index"].combine_chunks().to_numpy(zero_copy_only=False)
                )
                needs_write = original_rows != rows or not np.array_equal(current_index, expected_index)
                if needs_write:
                    column_index = table.schema.get_field_index("index")
                    index_type = table.schema.field(column_index).type
                    table = table.set_column(
                        column_index,
                        "index",
                        pa.array(expected_index, type=index_type),
                    )
                    write_parquet_atomic(path, table, compression)
                    rewritten += 1

                episode_row["length"] = rows
                episode_stats = update_stats(table, accumulators, stats_feature_names)
                stats_file.write(
                    json.dumps(
                        {"episode_index": episode_index, "stats": episode_stats},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                global_index += rows

                if position == 1 or position % 500 == 0 or position == total:
                    print(
                        f"  Parquet/stats [{position:0{width}d}/{total}] "
                        f"episode_{episode_index:06d}",
                        flush=True,
                    )

            stats_file.flush()
            os.fsync(stats_file.fileno())

        os.replace(stats_temporary, stats_path)
    except Exception:
        stats_temporary.unlink(missing_ok=True)
        raise

    info["total_frames"] = global_index
    image_feature_count = sum(
        value.get("dtype") == "image" for value in info["features"].values()
    )
    info["total_images"] = global_index * image_feature_count

    write_jsonl_atomic(dataset / "meta/episodes.jsonl", episode_rows)
    write_json_atomic(dataset / "meta/info.json", info)
    write_json_atomic(dataset / "meta/stats.json", finalize_global_stats(accumulators))
    return global_index, rewritten


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    dataset = args.dataset_root.resolve()
    if not dataset.is_dir():
        raise FileNotFoundError(dataset)

    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError("ffmpeg and ffprobe are required")

    roots = [metadata_root(path) for path in args.metadata_roots]
    specs = load_trim_specs(roots)
    info = json.loads((dataset / "meta/info.json").read_text(encoding="utf-8"))
    if info.get("codebase_version") != "v2.1":
        raise RuntimeError("Only LeRobot v2.1 is supported")

    episode_rows_list = load_jsonl(dataset / "meta/episodes.jsonl")
    episode_rows_list.sort(key=lambda row: int(row["episode_index"]))
    episode_rows = {int(row["episode_index"]): row for row in episode_rows_list}
    video_keys = sorted(
        key for key, value in info["features"].items() if value.get("dtype") == "video"
    )

    videos, depth_frames, already_trimmed = validate_media(
        dataset,
        info,
        specs,
        episode_rows,
        video_keys,
        ffprobe,
    )

    print("Padded tail trim")
    print(f"  Dataset:                  {dataset}")
    print(f"  Affected episodes:        {len(specs)}")
    print(f"  Already trimmed episodes: {already_trimmed}")
    print(f"  Videos to trim:           {len(videos)}")
    print(f"  Depth PNGs to delete:     {len(depth_frames)}")
    print(f"  Video mode:               H.264 stream copy (no re-encoding)")

    if args.dry_run:
        print("  Mode:                     dry run")
        print("Validation complete; no files were changed.")
        return

    for position, (path, final_length) in enumerate(videos, start=1):
        trim_video(path, final_length, ffmpeg, ffprobe)
        if position == 1 or position % 50 == 0 or position == len(videos):
            print(f"  Videos [{position}/{len(videos)}] {path}", flush=True)

    for path in depth_frames:
        path.unlink()
    print(f"  Deleted depth PNGs:       {len(depth_frames)}", flush=True)

    total_frames, rewritten = rebuild_parquet_and_metadata(
        dataset,
        info,
        episode_rows_list,
        specs,
    )

    print("Padded tail trim complete")
    print(f"  Final total frames:       {total_frames}")
    print(f"  Rewritten Parquet files:  {rewritten}")
    print(f"  Elapsed seconds:          {time.monotonic() - started:.1f}")


if __name__ == "__main__":
    main()
