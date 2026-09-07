#!/usr/bin/env python3
"""Rootless Linux x86_64 runtime; never touches system Python/CUDA/Conda."""
import argparse
import hashlib
import os
from pathlib import Path
import platform
import re
import subprocess
import tarfile
import time
import urllib.request


UV_VERSION = "0.8.22"
PYTHON_VERSION = "3.11.11"


def download(url, dest):
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=60) as response, dest.open("wb") as f:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    f.write(block)
            return
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work_dir", type=Path)
    profile = parser.add_mutually_exclusive_group()
    profile.add_argument("--gpu", action="store_true", help="Install CUDA wheels and test accessible GPUs")
    profile.add_argument("--prepare-gpu", action="store_true", help="Install CUDA wheels on a CPU host; test CPU operations only")
    profile.add_argument("--cpu-torch", action="store_true", help="Install CPU-only PyTorch for geometric checks")
    parser.add_argument("--verify-only", action="store_true", help="Verify the prepared runtime without installing or downloading")
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        parser.error("Runtime installer supports Linux x86_64 (including Debian 12).")
    root = args.work_dir.expanduser().resolve() / "runtime"
    root.mkdir(parents=True, exist_ok=True)
    # Serialize installation even when two hosts request status or launch the
    # same worker directory before the workflow-level lock has been acquired.
    import fcntl
    with (root / 'bootstrap.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        install(root,args.gpu,args.prepare_gpu,args.cpu_torch,args.verify_only)


def install(root,gpu=False,prepare_gpu=False,cpu_torch=False,verify_only=False):
    env = os.environ.copy()
    for name in ("PYTHONHOME", "PYTHONPATH", "LD_PRELOAD", "LD_LIBRARY_PATH"):
        env.pop(name, None)
    env.update(PYTHONNOUSERSITE="1", UV_PYTHON_INSTALL_DIR=str(root / "python"),
               UV_CACHE_DIR=str(root / "cache"), UV_PYTHON_PREFERENCE="only-managed")
    uv = root / "uv"
    if not uv.exists():
        if verify_only:
            raise RuntimeError("Runtime not prepared; run bootstrap on the CPU host first")
        name = "uv-x86_64-unknown-linux-gnu.tar.gz"
        url = f"https://github.com/astral-sh/uv/releases/download/{UV_VERSION}/{name}"
        package, checksum = root / name, root / (name + ".sha256")
        download(url, package)
        download(url + ".sha256", checksum)
        actual = hashlib.sha256(package.read_bytes()).hexdigest()
        if actual != checksum.read_text().split()[0]:
            raise RuntimeError("uv release checksum mismatch")
        with tarfile.open(package) as tar:
            member = tar.getmember("uv-x86_64-unknown-linux-gnu/uv")
            uv.write_bytes(tar.extractfile(member).read())
        uv.chmod(0o755)
    def run(*cmd):
        subprocess.run([str(x) for x in cmd], check=True, env=env)
    version = subprocess.check_output([str(uv), "--version"], text=True).strip()
    if version.split()[:2] != ["uv", UV_VERSION]:
        raise RuntimeError(f"Unexpected uv binary: {version}")
    venv = root / "env"
    python = venv / "bin/python"
    if not python.exists():
        if verify_only:
            raise RuntimeError("Python environment not prepared; run bootstrap on the CPU host first")
        run(uv, "venv", "--python", PYTHON_VERSION, venv)
    name = "requirements-gpu.lock" if gpu or prepare_gpu else "requirements-cpu.lock" if cpu_torch else "requirements.lock"
    req = Path(__file__).with_name(name)
    if not req.exists():
        raise RuntimeError(f"Incomplete checkout: missing {req}")
    if not verify_only:
        run(uv, "pip", "install", "--python", python, "--only-binary", ":all:",
            "--require-hashes", "--index-strategy", "unsafe-best-match", "-r", req)
    else:
        # No resolver/network access on a scheduled GPU worker. Reject a stale
        # or wrong-profile environment instead of silently installing there.
        import json
        pins = dict(re.findall(r"^([A-Za-z0-9_.-]+)==([^\s;\\]+)",req.read_text(),re.M))
        run(python,"-c", """import importlib.metadata as m,json,platform,sys
bad = {}
for key, expected in json.loads(sys.argv[1]).items():
    try:
        actual = m.version(key)
    except m.PackageNotFoundError:
        actual = 'missing'
    if actual != expected:
        bad[key] = (expected, actual)
if platform.python_version() != sys.argv[2]:
    bad['python'] = (sys.argv[2], platform.python_version())
if bad:
    raise SystemExit(f'Prepared runtime differs from lock; prepare again on CPU: {bad}')
""",json.dumps(pins),PYTHON_VERSION)
    run(uv, "pip", "check", "--python", python)
    run(python, "-m", "recam_refine", "doctor", *( ["--gpu"] if gpu else ["--torch-cpu"] if cpu_torch or prepare_gpu else [] ))
    print(f"Ready: {python}", flush=True)


if __name__ == "__main__":
    main()
