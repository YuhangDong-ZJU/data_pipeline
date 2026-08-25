from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

import annotate_normals_normalcrafter as worker  # noqa: E402
import download_droid_rgb_inputs as downloader  # noqa: E402


class ChunkParsingTest(unittest.TestCase):
    def test_worker_chunk_forms(self) -> None:
        self.assertEqual(
            worker.normalized_chunk_selectors(["0,2", "5-7", "chunk-009"]),
            {
                "chunk-000",
                "chunk-002",
                "chunk-005",
                "chunk-006",
                "chunk-007",
                "chunk-009",
            },
        )

    def test_download_chunk_forms(self) -> None:
        self.assertEqual(downloader.parse_chunks("0,2,5-7"), [0, 2, 5, 6, 7])

    def test_nested_dataset_prefix(self) -> None:
        self.assertEqual(downloader.normalize_prefix("/real_world/droid/"), "real_world/droid")
        with self.assertRaises(ValueError):
            downloader.normalize_prefix("../droid")


class ResumeSchedulerTest(unittest.TestCase):
    def make_task(self, root: Path, index: int) -> worker.NormalTask:
        episode = f"episode_{index:06d}"
        output = root / "normal" / f"{episode}.mp4"
        metadata = root / "logs" / f"{episode}.json"
        return worker.NormalTask(
            subset_root=root,
            subset_relative=Path("."),
            chunk="chunk-000",
            input_camera="observation.images.rgb_01",
            output_camera="observation.images.normal_01",
            episode=episode,
            input_path=root / "rgb" / f"{episode}.mp4",
            output_path=output,
            metadata_path=metadata,
        )

    def mark_complete(self, task: worker.NormalTask) -> None:
        task.output_path.parent.mkdir(parents=True, exist_ok=True)
        task.output_path.write_bytes(b"atomic-video")
        task.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        task.metadata_path.write_text(
            json.dumps(
                {
                    "status": "complete",
                    "output_video": {"width": 1280, "height": 720, "frames": 10},
                }
            ),
            encoding="utf-8",
        )

    def test_completed_outputs_are_removed_before_resharding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks = [self.make_task(root, index) for index in range(5)]
            self.mark_complete(tasks[0])
            self.mark_complete(tasks[3])
            args = SimpleNamespace(overwrite=False, num_shards=2, shard_index=0, limit=None)

            selected, pending_count = worker.select_worker_tasks(tasks, args)

            self.assertEqual(pending_count, 3)
            self.assertEqual([task.episode for task in selected], [tasks[1].episode, tasks[4].episode])


if __name__ == "__main__":
    unittest.main()
