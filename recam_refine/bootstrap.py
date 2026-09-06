#!/usr/bin/env python3
"""Rootless Linux x86_64 runtime; never touches system Python/CUDA/Conda."""
import argparse
import hashlib
import os
from pathlib import Path
import platform
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
    parser.add_argument("--gpu", action="store_true", help="Install PyTorch CUDA 12.4 wheels for H100/4090")
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        parser.error("Runtime installer supports Linux x86_64 (including Debian 12).")
    root = args.work_dir.expanduser().resolve() / "runtime"
    root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    for name in ("PYTHONHOME", "PYTHONPATH", "LD_PRELOAD", "LD_LIBRARY_PATH"):
        env.pop(name, None)
    env.update(PYTHONNOUSERSITE="1", UV_PYTHON_INSTALL_DIR=str(root / "python"),
               UV_CACHE_DIR=str(root / "cache"), UV_PYTHON_PREFERENCE="only-managed")
    uv = root / "uv"
    if not uv.exists():
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
        run(uv, "venv", "--python", PYTHON_VERSION, venv)
    req = Path(__file__).with_name("requirements-gpu.lock" if args.gpu else "requirements.lock")
    if not req.exists():
        raise RuntimeError(f"Incomplete checkout: missing {req}")
    run(uv, "pip", "install", "--python", python, "--only-binary", ":all:",
        "--require-hashes", "--index-strategy", "unsafe-best-match", "-r", req)
    run(uv, "pip", "check", "--python", python)
    run(python, "-m", "recam_refine", "doctor", *( ["--gpu"] if args.gpu else [] ))
    print(f"Ready: {python}", flush=True)


if __name__ == "__main__":
    main()
