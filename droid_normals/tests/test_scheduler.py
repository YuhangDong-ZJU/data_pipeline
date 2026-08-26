from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


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


class OutputRecoveryTest(unittest.TestCase):
    def make_task(self, root: Path, index: int = 1) -> worker.NormalTask:
        episode = f"episode_{index:06d}"
        return worker.NormalTask(
            subset_root=root,
            subset_relative=Path("."),
            chunk="chunk-000",
            input_camera="observation.images.rgb_01",
            output_camera="observation.images.normal_01",
            episode=episode,
            input_path=root / "rgb" / f"{episode}.mp4",
            output_path=root / "normal" / f"{episode}.mp4",
            metadata_path=root / "logs" / f"{episode}.json",
        )

    def test_dead_same_host_lock_is_recovered_without_waiting_for_ttl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = self.make_task(Path(directory))
            lock_path = task.output_path.with_name(f".{task.output_path.name}.lock")
            lock_path.parent.mkdir(parents=True)
            lock_path.write_text(
                json.dumps(
                    {
                        "host": worker.socket.gethostname(),
                        "pid": 999999,
                        "created_unix": worker.time.time(),
                    }
                ),
                encoding="utf-8",
            )
            with patch.object(worker, "process_exists", return_value=False):
                lock = worker.OutputLock(task.output_path, stale_hours=0, heartbeat_seconds=0)
                self.assertTrue(lock.acquire())
            self.assertTrue(lock.recovered_stale_lock)
            lock.release()
            self.assertFalse(lock_path.exists())

    def test_live_owner_lock_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = self.make_task(Path(directory))
            first = worker.OutputLock(task.output_path, stale_hours=0, heartbeat_seconds=0)
            second = worker.OutputLock(task.output_path, stale_hours=0, heartbeat_seconds=0)
            self.assertTrue(first.acquire())
            self.assertFalse(second.acquire())
            first.release()

    def test_release_does_not_remove_a_replaced_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = self.make_task(Path(directory))
            lock = worker.OutputLock(task.output_path, stale_hours=0, heartbeat_seconds=0)
            self.assertTrue(lock.acquire())
            lock.path.write_text(json.dumps({"token": "another-owner"}), encoding="utf-8")
            lock.release()
            self.assertTrue(lock.path.exists())

    def test_orphan_parts_for_one_task_are_removed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = self.make_task(root)
            task.output_path.parent.mkdir(parents=True)
            task.metadata_path.parent.mkdir(parents=True)
            parts = [
                task.output_path.parent / f".{task.output_path.stem}.123.part.mp4",
                task.output_path.parent / f".{task.output_path.stem}.host.123.token.part.mp4",
                task.metadata_path.parent / f".{task.metadata_path.name}.123.part",
            ]
            for path in parts:
                path.write_bytes(b"partial")
            unrelated = task.output_path.parent / ".episode_000002.123.part.mp4"
            unrelated.write_bytes(b"keep")

            removed, removed_bytes = worker.cleanup_orphan_task_parts(task)

            self.assertEqual(removed, len(parts))
            self.assertEqual(removed_bytes, len(parts) * len(b"partial"))
            self.assertTrue(unrelated.exists())


class ModelRecoveryTest(unittest.TestCase):
    def test_reset_restores_fp16_and_clears_cuda_cache(self) -> None:
        class FakeModule:
            def __init__(self, dtype: str) -> None:
                self.dtype = dtype

            def to(self, *, dtype: str) -> None:
                self.dtype = dtype

        class FakeCuda:
            def __init__(self) -> None:
                self.empty_cache_calls = 0

            def empty_cache(self) -> None:
                self.empty_cache_calls += 1

        runner = worker.NormalCrafterRunner.__new__(worker.NormalCrafterRunner)
        cuda = FakeCuda()
        runner.torch = SimpleNamespace(cuda=cuda)
        runner.inference_dtype = "float16"
        runner.pipe = SimpleNamespace(vae=FakeModule("float32"), unet=FakeModule("float16"))

        restored = runner.reset_after_attempt()

        self.assertEqual(restored, ["vae"])
        self.assertEqual(runner.pipe.vae.dtype, "float16")
        self.assertEqual(cuda.empty_cache_calls, 2)


if __name__ == "__main__":
    unittest.main()
