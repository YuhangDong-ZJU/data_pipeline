from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

import annotate_normals_normalcrafter as worker  # noqa: E402


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


class HostMemoryTest(unittest.TestCase):
    def make_config_args(self) -> SimpleNamespace:
        return SimpleNamespace(
            unet_path=worker.DEFAULT_UNET_PATH,
            unet_revision=worker.DEFAULT_UNET_REVISION,
            pretrain_path=worker.DEFAULT_PRETRAIN_PATH,
            pretrain_revision=worker.DEFAULT_PRETRAIN_REVISION,
            target_fps=15,
            process_length=-1,
            max_res=1024,
            window_size=14,
            time_step_size=10,
            decode_chunk_size=7,
            seed=42,
            cpu_offload="none",
            output_width=1280,
            output_height=720,
            crf=17,
            ffmpeg_preset="veryfast",
        )

    def test_memory_controls_do_not_change_annotation_fingerprint(self) -> None:
        first = self.make_config_args()
        first.video_decode_batch_size = 16
        first.prefetch_next_video = False
        second = self.make_config_args()
        second.video_decode_batch_size = 64
        second.prefetch_next_video = True

        self.assertEqual(worker.annotation_config(first), worker.annotation_config(second))

    def test_video_is_decoded_in_bounded_batches(self) -> None:
        readers: list[object] = []

        class FakeBatch:
            def __init__(self, indexes: list[int], probe: bool) -> None:
                self.shape = (len(indexes), 720, 1280, 3) if probe else None
                self.array = worker.np.zeros(
                    (len(indexes), 2, 2, 3), dtype=worker.np.uint8
                )

            def asnumpy(self) -> worker.np.ndarray:
                return self.array

        class FakeVideoReader:
            def __init__(self, _path: str, **kwargs: object) -> None:
                self.kwargs = kwargs
                self.calls: list[list[int]] = []
                self.probe = "width" not in kwargs
                readers.append(self)

            def __len__(self) -> int:
                return 5

            def get_avg_fps(self) -> float:
                return 15.0

            def get_batch(self, indexes: list[int]) -> FakeBatch:
                self.calls.append(list(indexes))
                return FakeBatch(list(indexes), self.probe)

        class FakeImage:
            def copy(self) -> object:
                return object()

            def close(self) -> None:
                pass

        decord = types.ModuleType("decord")
        decord.VideoReader = FakeVideoReader
        decord.cpu = lambda index: index
        pil = types.ModuleType("PIL")
        pil.Image = SimpleNamespace(fromarray=lambda _frame: FakeImage())
        runner = worker.NormalCrafterRunner.__new__(worker.NormalCrafterRunner)
        runner.verbose_inference = False
        task = worker.NormalTask(
            subset_root=Path("."),
            subset_relative=Path("."),
            chunk="chunk-000",
            input_camera="observation.images.rgb_01",
            output_camera="observation.images.normal_01",
            episode="episode_000000",
            input_path=Path("episode_000000.mp4"),
            output_path=Path("normal.mp4"),
            metadata_path=Path("normal.json"),
        )
        args = SimpleNamespace(
            process_length=-1,
            target_fps=15,
            max_res=1024,
            video_decode_batch_size=2,
        )

        with patch.dict(sys.modules, {"decord": decord, "PIL": pil}):
            with patch.object(worker, "release_host_memory") as release:
                frames, fps = runner.load_video(task, args)

        self.assertEqual(len(frames), 5)
        self.assertEqual(fps, 15.0)
        self.assertEqual(readers[0].calls, [[0]])
        self.assertEqual(readers[1].calls, [[0, 1], [2, 3], [4]])
        self.assertEqual(readers[1].kwargs["width"], 1024)
        self.assertEqual(readers[1].kwargs["height"], 576)
        release.assert_called_once()

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
        task.output_path.write_bytes(b"\x00\x00\x00\x18ftypisomatomic-video")
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

    def test_stable_shard_is_selected_before_completed_outputs_are_removed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks = [self.make_task(root, index) for index in range(5)]
            self.mark_complete(tasks[0])
            self.mark_complete(tasks[3])
            args = SimpleNamespace(overwrite=False, num_shards=2, shard_index=0, limit=None)

            selected, pending_count = worker.select_worker_tasks(tasks, args)

            self.assertEqual(pending_count, 3)
            self.assertEqual([task.episode for task in selected], [tasks[2].episode, tasks[4].episode])

    def test_different_resume_views_do_not_shift_shard_ownership(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks = [self.make_task(root, index) for index in range(6)]
            shard_zero = SimpleNamespace(
                overwrite=False, num_shards=2, shard_index=0, limit=None
            )
            shard_one = SimpleNamespace(
                overwrite=False, num_shards=2, shard_index=1, limit=None
            )

            selected_zero, _ = worker.select_worker_tasks(tasks, shard_zero)
            self.mark_complete(tasks[0])
            selected_one, _ = worker.select_worker_tasks(tasks, shard_one)

            self.assertEqual(
                [task.episode for task in selected_zero],
                [tasks[0].episode, tasks[2].episode, tasks[4].episode],
            )
            self.assertEqual(
                [task.episode for task in selected_one],
                [tasks[1].episode, tasks[3].episode, tasks[5].episode],
            )


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

    def make_config(self) -> dict[str, object]:
        args = SimpleNamespace(
            unet_path=worker.DEFAULT_UNET_PATH,
            unet_revision=worker.DEFAULT_UNET_REVISION,
            pretrain_path=worker.DEFAULT_PRETRAIN_PATH,
            pretrain_revision=worker.DEFAULT_PRETRAIN_REVISION,
            target_fps=15,
            process_length=-1,
            max_res=1024,
            window_size=14,
            time_step_size=10,
            decode_chunk_size=7,
            seed=42,
            cpu_offload="none",
            output_width=1280,
            output_height=720,
            crf=17,
            ffmpeg_preset="veryfast",
        )
        return worker.annotation_config(args)

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

    def test_non_mp4_output_is_not_complete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = self.make_task(Path(directory))
            task.output_path.parent.mkdir(parents=True)
            task.output_path.write_bytes(b"not an mp4")
            task.metadata_path.parent.mkdir(parents=True)
            task.metadata_path.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "output_video": {"width": 1280, "height": 720, "frames": 10},
                    }
                ),
                encoding="utf-8",
            )

            self.assertFalse(worker.task_is_complete(task))

    def test_config_fingerprint_mismatch_is_not_complete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = self.make_task(Path(directory))
            task.output_path.parent.mkdir(parents=True)
            task.output_path.write_bytes(b"\x00\x00\x00\x18ftypisomvideo")
            task.metadata_path.parent.mkdir(parents=True)
            config = self.make_config()
            task.metadata_path.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "config_fingerprint": worker.config_fingerprint(config),
                        "output_video": {"width": 1280, "height": 720, "frames": 10},
                    }
                ),
                encoding="utf-8",
            )

            self.assertTrue(worker.task_is_complete(task, config))
            changed_config = dict(config)
            changed_config["settings"] = dict(config["settings"], max_res=720)
            self.assertFalse(worker.task_is_complete(task, changed_config))

    def test_matching_legacy_config_remains_complete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = self.make_task(Path(directory))
            task.output_path.parent.mkdir(parents=True)
            task.output_path.write_bytes(b"\x00\x00\x00\x18ftypisomlegacy-video")
            task.metadata_path.parent.mkdir(parents=True)
            config = self.make_config()
            settings = config["settings"]
            task.metadata_path.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "method": "NormalCrafter",
                        "normalcrafter_commit": worker.NORMALCRAFTER_COMMIT,
                        "settings": {
                            key: settings[key]
                            for key in (
                                "target_fps",
                                "process_length",
                                "max_res",
                                "window_size",
                                "time_step_size",
                                "decode_chunk_size",
                                "seed",
                                "cpu_offload",
                                "crf",
                                "pixel_format",
                            )
                        },
                        "output_video": {"width": 1280, "height": 720, "frames": 10},
                    }
                ),
                encoding="utf-8",
            )

            self.assertTrue(worker.task_is_complete(task, config))


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

    def test_successful_attempt_keeps_cuda_allocator_cache(self) -> None:
        class FakeModule:
            dtype = "float16"

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
        runner.pipe = SimpleNamespace(vae=FakeModule(), unet=FakeModule())

        restored = runner.reset_after_attempt()

        self.assertEqual(restored, [])
        self.assertEqual(cuda.empty_cache_calls, 0)


if __name__ == "__main__":
    unittest.main()
