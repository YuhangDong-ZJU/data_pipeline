"""Validate TAR paths and bytes before installing depth PNGs."""
from __future__ import annotations

import hashlib
import os
import re
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


def check_png(path, shape=None, depth=True):
    with Image.open(path) as im:
        require(im.format == "PNG", f"Not PNG: {path}")
        if shape:
            require(im.size == (shape[1], shape[0]), f"PNG resolution mismatch: {path}")
        if depth:
            require(im.mode in ("I;16", "I;16L", "I"), f"Depth is not uint16 grayscale: {path}: {im.mode}")
            with Path(path).open("rb") as f:
                header = f.read(26)
            require(len(header) == 26 and header[24:26] == bytes([16, 0]), f"Depth PNG must be 16-bit grayscale: {path}")
        im.verify()


def unpack_archive(archive, subset, receipt_root):
    archive, subset = Path(archive), Path(subset)
    rel_archive = archive.relative_to(subset).as_posix()
    receipt = Path(receipt_root) / (hashlib.sha256(str(archive).encode()).hexdigest() + ".json")
    # Receipts are consumed only before any trimming starts.
    archive_hash = sha256(archive)
    if receipt.exists():
        saved = read_json(receipt)
        require(saved["sha256"] == archive_hash, f"Archive changed since extraction: {archive}")
    members = set()
    bytes_written = 0
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
            target = safe_path(subset, rel)
            target.parent.mkdir(parents=True, exist_ok=True)
            part = target.with_name("." + target.name + ".unpack-part")
            h = hashlib.sha256()
            size = 0
            try:
                with tar.extractfile(member) as src, part.open("wb") as dst:
                    for block in iter(lambda: src.read(1024 * 1024), b""):
                        dst.write(block)
                        h.update(block)
                        size += len(block)
                    dst.flush()
                    os.fsync(dst.fileno())
                require(size == member.size, f"Truncated TAR member: {rel}")
                check_png(part)
                if target.exists():
                    require(sha256(target) == h.hexdigest(), f"Existing PNG conflicts with TAR: {target}")
                else:
                    os.replace(part, target)
                    sync_dir(target.parent)
                    bytes_written += size
            finally:
                part.unlink(missing_ok=True)
    require(members, f"Empty depth TAR: {archive}")
    write_json(receipt, dict(archive=str(archive), sha256=archive_hash, files=len(members), bytes_written=bytes_written))
    return dict(archive=str(archive), sha256=archive_hash, files=len(members))


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
