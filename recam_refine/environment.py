"""Discover and run existing environments without installing any packages."""
from __future__ import annotations

import argparse
from contextlib import ExitStack, redirect_stdout
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


PROFILES = ('base', 'overlap', 'geometry', 'cpu', 'gpu', 'prepare-gpu')
BASE = {'numpy': 'numpy', 'pyarrow': 'pyarrow', 'Pillow': 'PIL', 'av': 'av',
        'scipy': 'scipy', 'huggingface-hub': 'huggingface_hub'}
GEOMETRY = {'trimesh': 'trimesh', 'pycollada': 'collada', 'networkx': 'networkx'}


def environment(python, cache):
    env = os.environ.copy()
    for key in ('PYTHONHOME', 'PYTHONPATH'):
        env.pop(key, None)
    env.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
               PATH=str(Path(python).parent) + os.pathsep + env.get('PATH', ''),
               XDG_CACHE_HOME=str(cache / 'xdg'), MPLCONFIGDIR=str(cache / 'matplotlib'),
               CUDA_CACHE_PATH=str(cache / 'cuda'), TRITON_CACHE_DIR=str(cache / 'triton'),
               TORCHINDUCTOR_CACHE_DIR=str(cache / 'torchinductor'),
               TORCH_EXTENSIONS_DIR=str(cache / 'torch_extensions'))
    # Keep the cluster's proxy, CA and dynamic-library configuration. CUDA
    # wheels can also run CPU checks on machines without an NVIDIA driver.
    return env


def probe(profile):
    """Offline capability checks, performed by the candidate interpreter."""
    import importlib
    import importlib.metadata as metadata
    import platform

    if sys.version_info[:2] not in ((3, 10), (3, 11)):
        raise RuntimeError('Supported existing environments use Python 3.10 or 3.11')
    packages = dict(BASE)
    if profile != 'base':
        packages['zstandard'] = 'zstandard'
    if profile in ('geometry', 'cpu', 'gpu', 'prepare-gpu'):
        packages.update(GEOMETRY)
    if profile == 'cpu':
        packages['matplotlib'] = 'matplotlib'
    if profile in ('cpu', 'gpu', 'prepare-gpu'):
        packages['torch'] = 'torch'
    versions, errors = {}, []
    for name, module in packages.items():
        try:
            versions[name] = metadata.version(name)
            importlib.import_module(module)
        except Exception as exc:
            errors.append(f'{name}: {type(exc).__name__}: {exc}')
    if errors:
        raise RuntimeError('Missing or unusable dependencies: ' + '; '.join(errors))
    if 'torch' in versions:
        allowed = ('2.8.0+cu129',) if profile in ('gpu', 'prepare-gpu') else ('2.8.0+cpu', '2.8.0+cu129')
        if versions['torch'] not in allowed:
            raise RuntimeError(f'Preserve installed torch {versions["torch"]}; this profile requires {allowed}')
    from recam_refine.__main__ import doctor
    with redirect_stdout(io.StringIO()) as checks:
        doctor(gpu=profile == 'gpu', torch_cpu=profile in ('cpu', 'prepare-gpu'))
    # Exercise the APIs actually used by PNG/media/Parquet processing. All
    # bytes are in memory; no dataset or environment files are modified.
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import av
    from PIL import Image
    from scipy.spatial.transform import Rotation
    pixels = np.array([[0, 65535], [1, 1234]], dtype=np.uint16)
    png = io.BytesIO()
    Image.fromarray(pixels).save(png, format='PNG')
    png.seek(0)
    if not np.array_equal(np.asarray(Image.open(png)), pixels):
        raise RuntimeError('uint16 depth PNG round trip failed')
    table = pa.table({'frames': [0, 1], 'pose': [[1., 2.], [3., 4.]]})
    parquet = io.BytesIO()
    pq.write_table(table, parquet)
    parquet.seek(0)
    if not pq.read_table(parquet).equals(table):
        raise RuntimeError('Parquet round trip failed')
    video = io.BytesIO()
    with av.open(video, mode='w', format='mp4') as container:
        stream = container.add_stream('libx264', rate=30)
        stream.width, stream.height, stream.pix_fmt = 16, 16, 'yuv420p'
        frame = av.VideoFrame.from_ndarray(np.zeros((16, 16, 3), np.uint8), format='rgb24')
        frame.pict_type = av.video.frame.PictureType.NONE
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    video.seek(0)
    with av.open(video) as container:
        if len(list(container.decode(video=0))) != 1:
            raise RuntimeError('H.264 encode/decode failed')
    if not np.allclose(Rotation.from_euler('xyz', [0, 0, 0]).as_matrix(), np.eye(3)):
        raise RuntimeError('Rotation conversion failed')
    if 'trimesh' in packages:
        import trimesh
        mesh = trimesh.creation.box()
        a, _ = trimesh.sample.sample_surface(mesh, 8, seed=17)
        b, _ = trimesh.sample.sample_surface(mesh, 8, seed=17)
        if not np.array_equal(a, b):
            raise RuntimeError('Deterministic robot mesh sampling failed')
    return dict(python=os.path.abspath(sys.executable), prefix=sys.prefix,
                python_version=platform.python_version(), packages=versions, profile=profile,
                checks=json.loads(checks.getvalue()))


def conda_executable():
    for key in ('DATA_PIPELINE_CONDA_BIN', 'DROID_NORMALS_CONDA_BIN', 'DROID_DEPTH_CONDA_BIN'):
        if os.environ.get(key):
            return os.path.expanduser(os.environ[key])
    if os.environ.get('MINIFORGE_HOME'):
        return str(Path(os.environ['MINIFORGE_HOME']).expanduser() / 'bin/conda')
    found = shutil.which('conda')
    if found:
        return found
    for name in ('miniforge3', 'miniconda3'):
        path = Path.home() / name / 'bin/conda'
        if path.is_file():
            return str(path)
    return None


def candidates(work, profile, python=None, conda_env=None):
    # Do not resolve the interpreter symlink: doing so discards a venv's prefix.
    if python:
        return [os.path.abspath(os.path.expanduser(python))]
    paths = []
    conda = conda_executable()
    if conda:
        result = subprocess.run([conda, 'env', 'list', '--json'], capture_output=True, text=True, timeout=30)
        if result.returncode:
            raise RuntimeError(f'Cannot list existing Conda environments with {conda}: {result.stderr.strip()}')
        prefixes = [Path(p) for p in json.loads(result.stdout)['envs']]
        names = [conda_env] if conda_env else list(dict.fromkeys([
            os.environ.get('DROID_NORMALS_ENV_NAME', 'droid_normals') if profile in ('gpu', 'prepare-gpu') else
            os.environ.get('TRIM_ENV_NAME', 'recam_data_pipeline'),
            os.environ.get('DROID_NORMALS_ENV_NAME', 'droid_normals'),
            os.environ.get('TRIM_ENV_NAME', 'recam_data_pipeline'),
            os.environ.get('RECAM_DOWNLOAD_ENV_NAME', 'recam_download')]))
        for name in names:
            matches = [p for p in prefixes if p.name == name]
            if len(matches) > 1:
                raise RuntimeError(f'Multiple environments named {name}; select --python /absolute/path/bin/python')
            paths.extend(str(p / 'bin/python') for p in matches)
    if conda_env:
        if not paths:
            raise RuntimeError(f'Existing Conda environment not found: {conda_env}; nothing was installed')
        return paths
    paths.append(str(work / 'runtime/env/bin/python'))
    if os.environ.get('CONDA_PREFIX'):
        paths.append(str(Path(os.environ['CONDA_PREFIX']) / 'bin/python'))
    paths.extend(p for p in (shutil.which('python'), shutil.which('python3')) if p)
    return list(dict.fromkeys(os.path.abspath(p) for p in paths))


def select(work, profile, cache, python=None, conda_env=None, leases=None):
    failures = []
    for candidate in candidates(work, profile, python, conda_env):
        if not os.path.isfile(candidate) or not os.access(candidate, os.X_OK):
            failures.append(f'{candidate}: executable not found')
            continue
        try:
            with ExitStack() as trial:
                fds = ()
                runtime = Path(candidate).parent.parent.resolve().parent
                if (runtime / 'bootstrap.lock').is_file():
                    from recam_refine.bootstrap import runtime_lock
                    lock = trial.enter_context(runtime_lock(runtime, read_only=True))
                    fds = (lock.fileno(),)
                result = subprocess.run([candidate, '-m', 'recam_refine.environment', '--probe', profile],
                                        env=environment(candidate, cache), capture_output=True, text=True, timeout=180,
                                        pass_fds=fds)
                if result.returncode == 0:
                    report = json.loads(result.stdout.strip().splitlines()[-1])
                    if leases is not None:
                        leases.enter_context(trial.pop_all())
                    return candidate, report, fds if leases is not None else ()
                failures.append(f'{candidate}: {result.stderr.strip() or result.stdout.strip()}')
        except subprocess.TimeoutExpired:
            failures.append(f'{candidate}: environment check timed out')
        except RuntimeError as exc:
            failures.append(f'{candidate}: {exc}')
    raise RuntimeError('No compatible existing environment. No installation or upgrade was attempted.\n' +
                       '\n'.join(failures) + '\nSelect another installed environment with --python or --conda-env.')


def freeze_gpu(work, report):
    """Do not mix library versions between machines or resume checkpoints with new ones."""
    import fcntl
    state = work / 'existing_gpu_environment.json'
    value = {k: report[k] for k in ('python_version', 'packages')}
    with (work / 'existing_gpu_environment.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if state.exists() and json.loads(state.read_text()) != value:
            raise RuntimeError('GPU environment changed between workers/runs; use the original package versions')
        if not state.exists():
            temporary = state.with_suffix('.tmp')
            temporary.write_text(json.dumps(value, indent=2) + '\n')
            temporary.replace(state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('work_dir', type=Path, nargs='?')
    parser.add_argument('--profile', choices=PROFILES, default='base')
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument('--python', default=os.environ.get('RECAM_REFINE_PYTHON'))
    selector.add_argument('--conda-env', default=os.environ.get('RECAM_REFINE_ENV_NAME'))
    parser.add_argument('--cache-work-dir', type=Path)
    parser.add_argument('--probe', choices=PROFILES, help=argparse.SUPPRESS)
    parser.add_argument('--exec', dest='command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    try:
        if args.probe:
            print(json.dumps(probe(args.probe)))
            return 0
        if args.work_dir is None or args.command == [] or (args.python and args.conda_env):
            parser.error('Supply work_dir, at most one environment selector, and a nonempty --exec command')
        work = args.work_dir.expanduser().resolve()
        work.mkdir(parents=True, exist_ok=True)
        log_work = (args.cache_work_dir or work).expanduser().resolve()
        cache = log_work / 'runtime_cache'
        for folder in ('xdg', 'matplotlib', 'cuda', 'triton', 'torchinductor', 'torch_extensions'):
            (cache / folder).mkdir(parents=True, exist_ok=True)
        with ExitStack() as leases:
            python, report, fds = select(work, args.profile, cache, args.python, args.conda_env, leases)
            if args.profile == 'gpu':
                freeze_gpu(work, report)
            (cache / f'environment_{args.profile}.json').write_text(json.dumps(report, indent=2) + '\n')
            print(f'REUSING EXISTING ENVIRONMENT: {python}\nNo packages installed or upgraded.', flush=True)
            print(json.dumps(report, indent=2), flush=True)
            if args.command:
                return subprocess.run([python, *args.command], env=environment(python, cache), pass_fds=fds).returncode
        return 0
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
