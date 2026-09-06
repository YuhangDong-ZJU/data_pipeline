"""Episode identities and small public calibration/manifest downloads."""
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
import re
import tarfile

from .common import require, read_json, read_jsonl, write_json, write_jsonl, atomic_bytes, sha256


def download_manifest(work, chunks):
    """Step 1 downloads source identities without PointWorld cameras/assets."""
    from huggingface_hub import HfApi, hf_hub_download
    work = Path(work) / 'inputs'
    work.mkdir(parents=True,exist_ok=True)
    versions = work/'hub_revisions.json'
    revisions = read_json(versions) if versions.exists() else {}
    repo = 'Sponbebob4258/droid-24k-external-svo'
    if repo not in revisions:
        revisions[repo] = HfApi().dataset_info(repo).sha
        write_json(versions,revisions)
    rows = []
    for chunk in chunks:
        path = hf_hub_download(repo,f'manifests/chunks/chunk-{chunk:03d}.jsonl',repo_type='dataset',
                               revision=revisions[repo],local_dir=work/repo.split('/')[1])
        rows.extend(read_jsonl(path))
    output = work/'transfer_episode_manifest.jsonl'
    write_jsonl(output,rows)
    return output


def download_inputs(work, chunks):
    from huggingface_hub import HfApi, hf_hub_download
    import zstandard
    work = Path(work) / "inputs"
    work.mkdir(parents=True, exist_ok=True)
    versions = work / "hub_revisions.json"
    repos = ["Sponbebob4258/droid-24k-external-svo", "nvidia/PointWorld-DROID"]
    revisions = read_json(versions) if versions.exists() else {}
    for repo in repos:
        if repo not in revisions:
            revisions[repo] = HfApi().dataset_info(repo).sha
    write_json(versions, revisions)
    def fetch(repo, filename):
        return Path(hf_hub_download(repo, filename, repo_type="dataset", revision=revisions[repo],
                                   local_dir=work / repo.split("/")[1]))
    manifest = []
    for chunk in chunks:
        manifest.extend(read_jsonl(fetch(repos[0], f"manifests/chunks/chunk-{chunk:03d}.jsonl")))
    merged = work / "episode_manifest.jsonl"
    write_jsonl(merged, manifest)
    cameras = work / "pointworld_cameras"
    stamp = work / "pointworld_cameras.complete.json"
    if not stamp.exists():
        package = fetch(repos[1], "droid/cameras/package.tar.zst.part-0000")
        cameras.mkdir(parents=True, exist_ok=True)
        count = 0
        names = set()
        with package.open("rb") as raw, zstandard.ZstdDecompressor().stream_reader(raw) as stream, tarfile.open(fileobj=stream, mode="r|") as tar:
            for member in tar:
                p = PurePosixPath(member.name)
                require(not p.is_absolute() and ".." not in p.parts, f"Unsafe PointWorld archive path: {p}")
                if member.isdir():
                    continue
                require(member.isfile() and re.fullmatch(r"[A-Za-z0-9+_.-]+_cameras\.json", p.name),
                        f"Unexpected PointWorld archive member: {p}")
                require(p.name not in names, f"Duplicate PointWorld UUID: {p.name}")
                names.add(p.name)
                data = tar.extractfile(member).read()
                value = json.loads(data)
                require(value.get("uuid") == p.name.removesuffix("_cameras.json"), f"PointWorld UUID/name mismatch: {p}")
                atomic_bytes(cameras / p.name, data)
                count += 1
        require(count > 40000, f"Unexpectedly small PointWorld release: {count}")
        write_json(stamp, dict(files=count, archive_sha256=sha256(package), revision=revisions[repos[1]]))
    return merged, cameras


def canonical_manifest(path, episodes):
    path = Path(path)
    paths = sorted(path.glob("chunk-*.jsonl")) if path.is_dir() else [path]
    rows = {}
    for p in paths:
        for r in read_jsonl(p):
            i = int(r["episode_index"])
            require(i not in rows, f"Duplicate episode index in manifest: {i}")
            if "camera_serials" not in r:
                r["camera_serials"] = {role: str(v["serial"]) for role, v in r["cameras"].items()}
            require(re.fullmatch(r"[A-Za-z0-9+_.-]+", r["source_episode_id"]), f"Unsafe episode ID: {i}")
            for role in ("external_1", "external_2"):
                require(role in r["camera_serials"], f"Missing camera serial: {i}:{role}")
            rows[i] = r
    ids = {int(e["episode_index"]) for e in episodes}
    require(ids <= rows.keys(), f"Manifest misses episodes: {sorted(ids - rows.keys())[:20]}")
    selected = {i: rows[i] for i in ids}
    require(len({r["source_episode_id"] for r in selected.values()}) == len(selected), "Duplicated source UUID")
    for e in episodes:
        r = selected[e["episode_index"]]
        require(0 <= int(r["length"]) - int(e["length"]) <= 2,
                f"Manifest length/index mapping disagrees with current metadata: {e['episode_index']}")
        if "tasks" in r:
            require(r["tasks"] == e["tasks"], f"Task/identity mismatch: episode {e['episode_index']}")
    return selected


def load_depth_records(roots, manifest):
    records = {}
    roots = list(dict.fromkeys(Path(p).resolve() for p in roots))
    for root in roots:
        nested = root / "annotations/foundation_stereo_depth"
        if nested.is_dir():
            root = nested
        # Direct annotation root and standard conversion output are supported.
        for p in root.glob("chunk-*/observation.images.depth_*/episode_*.json"):
            d = read_json(p)
            src = d["source"]
            i = int(src["episode_index"])
            if i not in manifest:
                continue
            role = src["camera_role"]
            if role not in ("external_1", "external_2"):
                continue
            r = manifest[i]
            require(src.get("source_episode_id") == r["source_episode_id"], f"Depth UUID mismatch: {p}")
            require(str(src["camera_serial"]) == str(r["camera_serials"][role]), f"Depth serial mismatch: {p}")
            n, decoded, missing = int(src["frame_count"]), int(src["decoded_frame_count"]), int(src["tail_missing_count"])
            require(n == decoded + missing and 0 <= missing <= 2 and decoded > 0, f"Invalid tail record: {p}")
            require(src["missing_frame_indices"] == list(range(decoded, n)), f"Non-tail decode failure: {p}")
            timestamps = src["timestamps_ms"]
            require(len(timestamps) == n and all(t is None for t in timestamps[decoded:]), f"Invalid padded timestamps: {p}")
            require(all(t is not None for t in timestamps[:decoded]), f"Interior missing timestamp: {p}")
            require(all(a < b for a, b in zip(timestamps[:decoded-1], timestamps[1:decoded])), f"Unordered timestamps: {p}")
            require("FoundationStereo" in d["inference"]["method"], f"Wrong depth method: {p}")
            key = (i, int(role[-1]))
            value = dict(path=str(p), sha256=sha256(p), frame_count=n, decoded=decoded, missing=missing,
                         intrinsic=d["calibration"]["intrinsic"], camera_serial=str(src["camera_serial"]))
            if key in records:
                a = {k:v for k,v in records[key].items() if k not in ("path", "sha256")}
                b = {k:v for k,v in value.items() if k not in ("path", "sha256")}
                require(a == b, f"Conflicting depth sidecars: {p}")
            records[key] = value
    return records
