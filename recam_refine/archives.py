"""Validate TAR paths and bytes before installing depth PNGs."""
from __future__ import annotations

import hashlib
import io
import os
import re
import stat
import tarfile
from pathlib import Path, PurePosixPath

from PIL import Image

from .common import require, safe_path, sha256, sync_dir, write_json, read_json


PNG_PATH = re.compile(r"images/chunk-\d{3}/observation\.images\.depth_\d{2}/episode_\d{6}/frame_\d{6}\.png")


def member_path(name, archive_relative):
    require("\\" not in name, f"Backslash in TAR path: {name}")
    path = PurePosixPath(name)
    require(not path.is_absolute() and ".." not in path.parts, f"Unsafe TAR path: {name}")
    # pack_depth.py writes paths from the subset root. Also accept the common
    # camera-relative form, without guessing or stripping arbitrary prefixes.
    if path.parts and path.parts[0] == "images":
        rel = path.as_posix()
    else:
        rel = (PurePosixPath(archive_relative).parent / path).as_posix()
    require(PNG_PATH.fullmatch(rel), f"Unexpected depth TAR member: {name}")
    require(PurePosixPath(rel).parts[:3] == PurePosixPath(archive_relative).parts[:3],
            f"TAR writes into a different camera/chunk: {name}")
    return rel


def _check_png_bytes(data, label, shape, depth, pixels=False):
    with Image.open(io.BytesIO(data)) as im:
        require(im.format == "PNG", f"Not PNG: {label}")
        if shape:
            require(im.size == (shape[1], shape[0]), f"PNG resolution mismatch: {label}")
        if depth:
            require(im.mode in ("I;16", "I;16L", "I"), f"Depth is not uint16 grayscale: {label}: {im.mode}")
            require(len(data) >= 26 and data[24:26] == bytes([16, 0]), f"Depth PNG must be 16-bit grayscale: {label}")
        im.load()
        if pixels:
            import numpy as np
            return np.asarray(im)


def check_png(path, shape=None, depth=True):
    _check_png_bytes(Path(path).read_bytes(), path, shape, depth)


def checked_png_array(path, shape=None, depth=True):
    """Read and decode once, without a separate PNG verify/CRC pass."""
    data = Path(path).read_bytes()
    return _check_png_bytes(data, path, shape, depth, pixels=True)


def _file_state(path):
    try:
        s = path.lstat()
        return [s.st_dev, s.st_ino, s.st_mode, s.st_size, s.st_mtime_ns, s.st_ctime_ns]
    except FileNotFoundError:
        return None


def _same_state(a, b):
    # Device/inode numbers can differ between clients of the same shared mount.
    if a is None or b is None:
        return a is b
    return a[2:] == b[2:]


def _same_bytes(a, b):
    with a.open('rb') as first, b.open('rb') as second:
        while True:
            left = first.read(1024 * 1024)
            right = second.read(1024 * 1024)
            if left != right:
                return False
            if not left:
                return True


def unpack_archive(archive, subset, receipt_root, authoritative_streams=None):
    archive, subset = Path(archive), Path(subset)
    rel_archive = archive.relative_to(subset).as_posix()
    directories = {}
    def target_path(rel):
        # Validate a stream directory once, not once for every PNG on shared storage.
        require(PNG_PATH.fullmatch(rel), f'Unexpected depth path: {rel}')
        relative = PurePosixPath(rel)
        parent = relative.parent.as_posix()
        if parent not in directories:
            directories[parent] = safe_path(subset, parent)
        return directories[parent]/relative.name
    receipt = Path(receipt_root) / (hashlib.sha256(str(archive).encode()).hexdigest() + ".json")
    archive_state = _file_state(archive)
    repair = set()
    reusable = {}
    # Receipts are consumed only before any trimming starts.
    if receipt.exists():
        saved = read_json(receipt)
        if _same_state(saved.get('archive_state'), archive_state) and 'target_states' in saved:
            for rel, state in saved['target_states'].items():
                current = _file_state(target_path(rel))
                if saved.get('superseded_by_metric_depth'):
                    # Legacy overlapping archives must never restore obsolete depth.
                    require(_same_state(current, state), f'Extracted/migrated file changed: {rel}')
                valid = ((current is None and state is None) or
                         (current is not None and state is not None and
                          stat.S_ISREG(current[2]) and current[3] == state[3]))
                if valid:
                    reusable[rel] = current
                else:
                    repair.add(rel)
            if not repair:
                print(f'SKIPPED 解压：已完成，文件存在且大小一致 {archive}', flush=True)
                return {k:saved[k] for k in ('archive', 'sha256', 'files', 'superseded_by_metric_depth', 'archive_state')}
            print(f'REPAIR 解压：恢复 {len(repair)} 个缺失或大小不符的文件；其余 {len(reusable)} 个跳过 {archive}', flush=True)
        require((saved.get("archive_state") is None or _same_state(saved["archive_state"], archive_state)), f"Archive changed since extraction: {archive}")
    members = set()
    bytes_written = 0
    superseded = 0
    target_states = {}
    authoritative_streams = authoritative_streams or {}
    with tarfile.open(archive, "r:*") as tar:
        for member in tar:
            require(not (member.issym() or member.islnk() or member.isdev() or member.isfifo()),
                    f"Links/special files are forbidden in depth TAR: {member.name}")
            if member.isdir():
                p = PurePosixPath(member.name)
                require(not p.is_absolute() and ".." not in p.parts and "\\" not in member.name,
                        f"Unsafe TAR directory: {member.name}")
                continue
            require(member.isfile(), f"Unsupported TAR member: {member.name}")
            rel = member_path(member.name, rel_archive)
            require(rel not in members, f"Duplicate TAR member: {rel}")
            members.add(rel)
            target = target_path(rel)
            if repair and rel in reusable:
                target_states[rel] = reusable[rel]
                continue
            require(not target.is_symlink(), f'Symlink is not supported: {target}')
            authority = authoritative_streams.get(PurePosixPath(rel).parent.as_posix())
            if authority is not None:
                if target.name in authority:
                    expected = authority[target.name]
                    require(target.is_file() and not target.is_symlink() and
                            (expected['size'] is None or target.stat().st_size == expected['size'] if isinstance(expected, dict)
                             else True),
                            f'Migrated depth changed before TAR extraction: {target}')
                else:
                    require(not target.exists(), f'Unexpected frame outside migrated depth stream: {target}')
                # Old depth is superseded; do not extract/decode it only to discard it.
                superseded += 1
                target_states[rel] = _file_state(target)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            part = target.with_name("." + target.name + ".unpack-part")
            size = 0
            try:
                with tar.extractfile(member) as src, part.open("wb") as dst:
                    for block in iter(lambda: src.read(1024 * 1024), b""):
                        dst.write(block)
                        size += len(block)
                    dst.flush()
                    os.fsync(dst.fileno())
                require(size == member.size, f"Truncated TAR member: {rel}")
                if target.exists() and rel not in repair:
                    require(_same_bytes(target, part), f"Existing PNG conflicts with TAR: {target}")
                else:
                    os.replace(part, target)
                    sync_dir(target.parent)
                    bytes_written += size
            finally:
                part.unlink(missing_ok=True)
            target_states[rel] = _file_state(target)
    require(members, f"Empty depth TAR: {archive}")
    require(not (repair - members), f'Repair files absent from TAR: {sorted(repair-members)[:3]}')
    require(_file_state(archive) == archive_state, f'Archive changed during extraction: {archive}')
    write_json(receipt, dict(archive=str(archive), sha256=None, files=len(members),
                            bytes_written=bytes_written,superseded_by_metric_depth=superseded,
                            archive_state=archive_state, target_states=target_states))
    return dict(archive=str(archive), sha256=None, files=len(members),superseded_by_metric_depth=superseded, archive_state=archive_state)


def frame_files(directory):
    directory = Path(directory)
    require(directory.is_dir(), f"Missing image stream: {directory}")
    files = sorted(directory.glob("frame_*.png"))
    require(files and [p.name for p in files] == [f"frame_{i:06d}.png" for i in range(len(files))],
            f"Missing/noncontiguous image frames: {directory}")
    require(all(p.is_file() and not p.is_symlink() for p in files), f"Invalid image files: {directory}")
    file_set = set(files)
    extras = [p for p in directory.iterdir() if p not in file_set and not p.name.endswith(".refine-part")]
    require(not extras, f"Unexpected files inside image stream: {extras[:3]}")
    return files
