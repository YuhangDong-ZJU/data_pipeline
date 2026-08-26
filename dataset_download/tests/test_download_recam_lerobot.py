from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

import download_recam_lerobot as downloader  # noqa: E402


class SubsetParsingTest(unittest.TestCase):
    def test_normalizes_and_deduplicates_subsets(self) -> None:
        self.assertEqual(
            downloader.normalize_subsets(
                ["real_world/droid,simulation/libero", "real_world/droid"]
            ),
            ["real_world/droid", "simulation/libero"],
        )

    def test_rejects_unsafe_subset_paths(self) -> None:
        for value in ("../droid", "/real_world/droid", "real_world\\droid", "a/*"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                downloader.normalize_subset(value)

    def test_all_cannot_be_combined_with_a_subset(self) -> None:
        with self.assertRaises(ValueError):
            downloader.normalize_subsets(["all", "real_world/droid"])

    def test_allow_pattern_is_limited_to_one_subset(self) -> None:
        self.assertEqual(
            downloader.allow_patterns("real_world/droid"),
            ["real_world/droid/**"],
        )


class FakeApi:
    def __init__(self) -> None:
        self.tree = {
            None: ["real_world", "simulation", "misc"],
            "real_world": ["real_world/droid"],
            "real_world/droid": [
                "real_world/droid/data",
                "real_world/droid/meta",
                "real_world/droid/videos",
            ],
            "simulation": ["simulation/libero"],
            "simulation/libero": ["simulation/libero/data"],
            "misc": ["misc/not_a_dataset"],
            "misc/not_a_dataset": [],
        }

    def list_repo_tree(self, *, path_in_repo: str | None, **_kwargs: object):
        return [
            SimpleNamespace(type="directory", path=path)
            for path in self.tree[path_in_repo]
        ]


class SubsetDiscoveryTest(unittest.TestCase):
    def test_discovers_dataset_leaf_directories(self) -> None:
        self.assertEqual(
            downloader.discover_subsets(FakeApi(), "owner/repo", "commit"),
            ["real_world/droid", "simulation/libero"],
        )


class DownloadRetryTest(unittest.TestCase):
    def test_retry_reuses_local_destination_and_validates_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            calls: list[dict[str, object]] = []

            def fake_snapshot_download(**kwargs: object) -> str:
                calls.append(kwargs)
                if len(calls) == 1:
                    raise TimeoutError("temporary network failure")
                output = destination / "real_world" / "droid" / "meta" / "info.json"
                output.parent.mkdir(parents=True)
                output.write_text("{}", encoding="utf-8")
                return str(destination)

            delays: list[float] = []
            downloader.download_subset(
                fake_snapshot_download,
                repo_id="owner/repo",
                revision="commit",
                destination=destination,
                subset="real_world/droid",
                workers=2,
                max_attempts=3,
                retry_delay_seconds=5,
                sleep=delays.append,
            )

            self.assertEqual(len(calls), 2)
            self.assertEqual(delays, [5])
            self.assertEqual(calls[1]["allow_patterns"], ["real_world/droid/**"])
            self.assertEqual(calls[1]["local_dir"], destination)
            self.assertEqual(calls[1]["revision"], "commit")


if __name__ == "__main__":
    unittest.main()
