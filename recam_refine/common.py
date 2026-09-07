"""Filesystem and schema primitives. All mutable state lives outside the dataset."""
from __future__ import annotations

import hashlib
import errno
import json
import os
import shutil
import re
from pathlib import Path

import numpy as np
import pyarrow as pa


class RefineError(RuntimeError):
    pass


def lock_mount(path):
    """Identify Linux mount semantics without invoking mount or changing options."""
    table = Path('/proc/self/mountinfo')
    if not table.exists():
        return '',set()
    target = Path(path).resolve()
    matches = []
    for line in table.read_text().splitlines():
        left,right = line.split(' - ',1)
        fields,details = left.split(),right.split()
        mount = Path(re.sub(r'\\([0-7]{3})',lambda m:chr(int(m[1],8)),fields[4]))
        if target.is_relative_to(mount):
            matches.append((len(mount.parts),details[0],set((fields[5]+','+details[2]).split(','))))
    return max(matches,key=lambda v:v[0])[1:] if matches else ('',set())


def validate_lock_mount(path):
    kind,options = lock_mount(path)
    if kind in ('nfs','nfs4'):
        require(not options&{'nolock','local_lock=all','local_lock=flock'},
                'Shared NFS locking is disabled by mount options; enable server-side flock for dataset/coordinator before running')
    require(kind not in ('fuse.sshfs','fuse.rclone','fuse.s3fs'),
            f'{kind} does not provide the shared POSIX locking required by this workflow')
    return kind


def acquire_directory_lock(fd,mode):
    """Additional dataset protection; caller MUST already hold work/run.lock.

    NFS emulates flock with byte-range locks and cannot exclusively lock a
    read-only directory. Shared coordinator locks are ordinary files opened
    for writing, and remain authoritative on network mounts.
    https://man7.org/linux/man-pages/man2/flock.2.html (NFS details)
    """
    import fcntl
    path = Path(os.readlink(f'/proc/self/fd/{fd}'))
    kind = validate_lock_mount(path)
    try:
        fcntl.flock(fd,mode|fcntl.LOCK_NB)
    except OSError as exc:
        if kind not in ('nfs','nfs4','cifs','smb3') or exc.errno not in (errno.EBADF,errno.EINVAL,errno.EISDIR,errno.EOPNOTSUPP):
            raise
        # No inode is added to the dataset, and no file being replaced is used
        # as a lock. Every machine must use the SAME shared coordinator.


def require(condition, message):
    if not condition:
        raise RefineError(str(message))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sync_dir(path):
    if os.name == "posix":
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name("." + path.name + ".refine-part")
    with part.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(part, path)
    sync_dir(path.parent)


def write_json(path, value):
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode())


def write_jsonl(path, rows):
    atomic_bytes(path, "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows).encode())


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def safe_path(root, relative):
    root = Path(root).resolve()
    relative = Path(relative)
    require(not relative.is_absolute() and ".." not in relative.parts, f"Unsafe path: {relative}")
    result = root / relative
    require(result.resolve().is_relative_to(root), f"Path escapes root: {result}")
    for p in [result, *result.parents]:
        if p == root:
            break
        require(not p.is_symlink(), f"Symlink is not supported: {p}")
    return result


def media_path(root, info, episode, key, frame=None):
    fields = dict(episode_chunk=episode // info["chunks_size"], episode_index=episode,
                  video_key=key, image_key=key, frame_index=frame)
    template = info["image_path" if frame is not None else "video_path"]
    return safe_path(root, template.format(**fields))


def parquet_path(root, info, episode):
    return safe_path(root, info["data_path"].format(episode_chunk=episode // info["chunks_size"], episode_index=episode))


def values(column):
    require(column.null_count == 0, "Nulls in a numerical column")
    arr = column.combine_chunks()
    shape = []
    while pa.types.is_fixed_size_list(arr.type):
        shape.append(arr.type.list_size)
        arr = arr.values
        require(arr.null_count == 0, "Nested nulls in a numerical column")
    require(pa.types.is_integer(arr.type) or pa.types.is_floating(arr.type) or pa.types.is_boolean(arr.type),
            f"Unsupported numerical type: {arr.type}")
    return np.asarray(arr.to_numpy(zero_copy_only=False)).reshape(len(column), *(shape or [1]))


def set_values(table, key, array):
    i = table.schema.get_field_index(key)
    require(i >= 0, f"Missing column: {key}")
    field = table.schema.field(i)
    if not pa.types.is_nested(field.type):
        array = np.asarray(array).reshape(-1)
    return table.set_column(i, field, pa.array(array.tolist(), type=field.type))


def array_hash(array):
    a = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def preserved_hashes(table):
    excluded = {"index", "observation.camera.extrinsics", "observation.camera.intrinsics"}
    return {key:array_hash(values(table[key])) for key in table.column_names if key not in excluded}


def check_transform(m, label):
    m = np.asarray(m, dtype=np.float64)
    require(m.shape[-2:] == (4, 4) and np.isfinite(m).all(), f"Invalid transform: {label}")
    require(np.allclose(m[..., 3, :], [0, 0, 0, 1], atol=2e-5), f"Invalid homogeneous row: {label}")
    r = m[..., :3, :3]
    require(np.allclose(r @ np.swapaxes(r, -1, -2), np.eye(3), atol=2e-3), f"Non-orthogonal rotation: {label}")
    require(np.allclose(np.linalg.det(r), 1, atol=2e-3), f"Reflection/non-unit rotation: {label}")
    return m


class Journal:
    """Back up before replacing; replaying an unfinished stage is idempotent.

    The dataset must be offline until SUCCESS.json is written. An interrupted
    multi-file update is resumed from the immutable plan, never re-planned.
    """
    def __init__(self, dataset, work):
        self.dataset = Path(dataset).resolve()
        self.work = Path(work).resolve()

    def backup(self, path):
        path = Path(path)
        rel = path.relative_to(self.dataset)
        saved = safe_path(self.work / "original", rel)
        if path.exists() and not saved.exists():
            saved.parent.mkdir(parents=True, exist_ok=True)
            part = saved.with_name(saved.name + ".part")
            shutil.copy2(path, part)
            require(sha256(part) == sha256(path), f"Backup verification failed: {path}")
            with part.open("rb") as f:
                os.fsync(f.fileno())
            os.replace(part, saved)
            sync_dir(saved.parent)
        return saved

    def replace(self, path, staged):
        self.backup(path)
        with Path(staged).open("rb") as f:
            os.fsync(f.fileno())
        os.replace(staged, path)
        sync_dir(Path(path).parent)

    def json(self, path, value, lines=False):
        self.backup(path)
        (write_jsonl if lines else write_json)(path, value)

    def retire(self, path, category="retired"):
        path = Path(path)
        if not path.exists():
            return
        require(not path.is_symlink(), f"Refusing to retire a symlink: {path}")
        relative = path.relative_to(self.dataset)
        dest = safe_path(self.work / category, relative)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            require(path.is_file() and dest.is_file() and sha256(path) == sha256(dest), f"Retirement conflict: {path}")
            path.unlink()
        else:
            # shutil.move supports separate mounts; copy is verified for files.
            if path.is_file() and path.stat().st_dev != dest.parent.stat().st_dev:
                part = dest.with_name(dest.name + ".part")
                shutil.copy2(path, part)
                require(sha256(part) == sha256(path), f"Move verification failed: {path}")
                os.replace(part, dest)
                path.unlink()
            else:
                shutil.move(str(path), str(dest))
        sync_dir(path.parent)
