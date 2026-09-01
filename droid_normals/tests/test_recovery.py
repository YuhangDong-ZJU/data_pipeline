from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

import normal_task_recovery as recovery  # noqa: E402


class NativeCrashRecoveryTest(unittest.TestCase):
    def identity(self) -> dict[str, str]:
        return {
            "subset": "real_world/droid",
            "chunk": "chunk-003",
            "input_camera": "observation.images.rgb_01",
            "output_camera": "observation.images.normal_01",
            "episode": "episode_003122",
            "input_path": "/dataset/rgb/episode_003122.mp4",
            "output_path": "/dataset/normal/episode_003122.mp4",
            "metadata_path": "/logs/episode_003122.json",
        }

    def test_native_crash_is_retried_then_quarantined(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state_dir = Path(directory) / "recovery"
            active = state_dir / "active" / "shard-0.json"
            identity = self.identity()

            for attempt in range(1, 4):
                with patch.object(recovery.os, "getpid", return_value=1234):
                    recovery.write_active_task(active, identity)
                result = recovery.record_native_crash(
                    state_dir,
                    active,
                    max_attempts=3,
                    exit_status=139,
                    gpu_id="0",
                    shard_index=0,
                    worker_pid=1234,
                )
                self.assertIsNotNone(result)
                assert result is not None
                self.assertEqual(result["native_crash_attempts"], attempt)
                expected = "quarantined" if attempt == 3 else "retrying"
                self.assertEqual(result["status"], expected)
                self.assertFalse(active.exists())

            key = recovery.task_key(identity)
            self.assertEqual(recovery.load_quarantined_keys(state_dir), {key})
            self.assertEqual(len(recovery.unresolved_quarantines(state_dir)), 1)

            recovery.mark_resolved(state_dir, key)
            self.assertEqual(recovery.load_quarantined_keys(state_dir), set())

    def test_wrong_worker_pid_does_not_charge_a_task(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state_dir = Path(directory) / "recovery"
            active = state_dir / "active" / "shard-0.json"
            with patch.object(recovery.os, "getpid", return_value=111):
                recovery.write_active_task(active, self.identity())

            result = recovery.record_native_crash(
                state_dir,
                active,
                max_attempts=3,
                exit_status=139,
                gpu_id="0",
                shard_index=0,
                worker_pid=222,
            )

            self.assertIsNone(result)
            self.assertTrue(active.exists())
            self.assertEqual(recovery.load_quarantined_keys(state_dir), set())

    def test_exhausted_python_retries_are_quarantined(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state_dir = Path(directory) / "recovery"
            payload = recovery.quarantine_python_failure(
                state_dir,
                self.identity(),
                attempts=3,
                error_type="RuntimeError",
                message="decoder failed",
            )

            self.assertEqual(payload["status"], "quarantined")
            self.assertEqual(payload["python_attempts"], 3)
            self.assertEqual(
                recovery.load_quarantined_keys(state_dir),
                {recovery.task_key(self.identity())},
            )


if __name__ == "__main__":
    unittest.main()
