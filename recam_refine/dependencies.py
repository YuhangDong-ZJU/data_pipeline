"""Add missing wheels to an existing environment while freezing installed packages."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
import urllib.request


PIP_URL = ('https://files.pythonhosted.org/packages/b7/3f/'
           '945ef7ab14dc4f9d7f40288d2df998d1837ee0888ec3659c813487572faa/pip-25.2-py3-none-any.whl')
PIP_SHA256 = '6d67a2b4e7f14d8b31b8b52648866fa717f45a1eb70e83002f4331d07e953717'


def canonical(name):
    return re.sub(r'[-_.]+', '-', name).lower()


def inventory(python, env, packages=()):
    code = '''import importlib.metadata as m,json,sys,sysconfig
versions = {}
for d in m.distributions():
    name = d.metadata.get('Name')
    if name:
        import re
        versions.setdefault(re.sub(r'[-_.]+', '-', name).lower(), d.version)
missing = set()
try:
    from packaging.requirements import Requirement
except ImportError:
    try:
        from pip._vendor.packaging.requirements import Requirement
    except ImportError:
        Requirement = None
if Requirement:
    queue = [(n, frozenset()) for n in json.loads(sys.argv[1])]
    seen = set()
    while queue:
        name, extras = queue.pop()
        key = (re.sub(r'[-_.]+', '-', name).lower(), extras)
        if key in seen:
            continue
        seen.add(key)
        try:
            requirements = m.requires(name) or []
        except m.PackageNotFoundError:
            missing.add(key[0])
            continue
        for line in requirements:
            req = Requirement(line)
            if req.marker is None or any(req.marker.evaluate({'extra': e}) for e in (extras or {''})):
                queue.append((req.name, frozenset(req.extras)))
print(json.dumps(dict(prefix=sys.prefix, base_prefix=sys.base_prefix,
    python_version=list(sys.version_info[:2]), purelib=sysconfig.get_path('purelib'), packages=versions,
    missing_transitive=sorted(missing))))
'''
    result = subprocess.run([python, '-c', code, json.dumps(list(packages))], env=env,
                            capture_output=True, text=True, check=True, timeout=30)
    return json.loads(result.stdout)


def missing_requirements(snapshot, packages, profile):
    if snapshot['python_version'] not in ([3, 10], [3, 11]):
        raise RuntimeError('Existing Python must be 3.10 or 3.11; it will not be replaced')
    installed = snapshot['packages']
    if 'torch' in packages and 'torch' in installed:
        allowed = ('2.8.0+cu129',) if profile in ('gpu', 'prepare-gpu') else ('2.8.0+cpu', '2.8.0+cu129')
        if installed['torch'] not in allowed:
            raise RuntimeError(f'Installed torch {installed["torch"]} is preserved; required: {allowed}')
    return [canonical(name) for name in packages if canonical(name) not in installed]


@contextmanager
def environment_lease(work, prefix, exclusive=False, timeout=30):
    """All workers sharing a coordinator serialize repairs and keep read leases."""
    import fcntl
    folder = work / 'existing_environment_locks'
    folder.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(str(Path(prefix).resolve()).encode()).hexdigest()
    with (folder / (key + '.lock')).open('a+') as lock:
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(lock, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)
                break
            except BlockingIOError as exc:
                if time.monotonic() >= deadline:
                    raise RuntimeError('Environment is in use or being supplemented; finish active workers before '
                                       'changing dependencies, then retry the same command') from exc
                time.sleep(.1)
        yield lock


def fetch_wheel(url, digest, path):
    if not re.fullmatch('[0-9a-f]{64}', digest):
        raise RuntimeError('Wheel must have a SHA-256 digest')
    if path.exists():
        sha = hashlib.sha256()
        with path.open('rb') as stream:
            while block := stream.read(1024 * 1024):
                sha.update(block)
        if sha.hexdigest() == digest:
            return
    import ssl
    ca = next((os.environ[n] for n in ('PIP_CERT', 'REQUESTS_CA_BUNDLE', 'SSL_CERT_FILE')
               if os.environ.get(n)), None)
    context = ssl.create_default_context(cafile=ca)
    temporary = path.with_suffix('.part')
    for attempt in range(4):
        try:
            sha = hashlib.sha256()
            with urllib.request.urlopen(url, timeout=60, context=context) as source, temporary.open('wb') as dest:
                while block := source.read(1024 * 1024):
                    dest.write(block)
                    sha.update(block)
            if sha.hexdigest() != digest:
                raise RuntimeError('Downloaded wheel SHA-256 mismatch')
            temporary.replace(path)
            return
        except (OSError, TimeoutError):
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def installer_environment(env, cache, python=None):
    result = dict(env)
    # Honor cluster network settings, but prevent pip configuration from sending
    # an install to --user, --target or another prefix or replacing old packages.
    network = {'PIP_INDEX_URL', 'PIP_EXTRA_INDEX_URL', 'PIP_FIND_LINKS', 'PIP_NO_INDEX',
               'PIP_CERT', 'PIP_CLIENT_CERT', 'PIP_PROXY', 'PIP_TIMEOUT', 'PIP_RETRIES'}
    for key in list(result):
        if key.startswith('PIP_') and key not in network:
            result.pop(key)
    # We discard pip's config file below so a stray install/target/prefix setting
    # cannot redirect the installation. That also drops the cluster mirror's
    # index-url, so read it from the effective config first and carry it forward
    # as PIP_INDEX_URL; supplement() relies on it as the general-index fallback.
    if python and 'PIP_INDEX_URL' not in result:
        try:
            probe = subprocess.run([python, '-m', 'pip', 'config', 'get', 'global.index-url'],
                                   env=env, text=True, capture_output=True, timeout=30)
            url = probe.stdout.strip()
            if probe.returncode == 0 and re.match(r'https?://', url):
                result['PIP_INDEX_URL'] = url
        except (OSError, subprocess.SubprocessError):
            pass
    result.update(PIP_CONFIG_FILE=os.devnull, PIP_DISABLE_PIP_VERSION_CHECK='1', PIP_NO_INPUT='1',
                  PIP_CACHE_DIR=str(cache / 'pip'), PIP_PROGRESS_BAR='off')
    return result


def pip_command(python, snapshot, cache):
    version = snapshot['packages'].get('pip', '0')
    parts = re.match(r'(\d+)\.(\d+)', version)
    if parts and tuple(map(int, parts.groups())) >= (22, 2):
        return [python, '-m', 'pip']
    # Run a pinned pure-Python wheel from the work directory if pip is absent or
    # too old for installation reports. Do not upgrade pip in the selected env.
    wheel = cache / 'pip-25.2-py3-none-any.whl'
    fetch_wheel(PIP_URL, PIP_SHA256, wheel)
    return [python, '-c', "import sys,runpy; sys.path.insert(0,sys.argv.pop(1)); "
            "runpy.run_module('pip',run_name='__main__')", str(wheel)]


def plan_requirements(snapshot, packages, profile):
    installed = snapshot['packages']
    pins = {canonical(n): v for n, v in re.findall(r'^([\w.-]+)==([^\s]+)',
            Path(__file__).with_name('requirements.txt').read_text(), re.M)}
    pins['torch'] = '2.8.0+cpu' if profile == 'cpu' else '2.8.0+cu129'
    # Existing versions are requirements as well as constraints so pip resolves
    # their missing transitive dependencies without upgrading the root package.
    return [f'{canonical(name)}=={installed.get(canonical(name), pins[canonical(name)])}' for name in packages]


def validate_plan(plan, installed):
    rows = plan.get('install')
    if not isinstance(rows, list):
        raise RuntimeError('Invalid pip installation report')
    seen = set()
    for row in rows:
        name = canonical(row['metadata']['name'])
        if name in installed:
            raise RuntimeError(f'Refusing to replace already-installed {name}=={installed[name]}')
        if name in seen:
            raise RuntimeError(f'Duplicate wheel in installation plan: {name}')
        seen.add(name)
        url = row['download_info']['url']
        from urllib.parse import urlparse, unquote
        filename = Path(unquote(urlparse(url).path)).name
        sha = row['download_info'].get('archive_info', {}).get('hashes', {}).get('sha256', '')
        if not filename.endswith('.whl') or not re.fullmatch('[0-9a-f]{64}', sha):
            raise RuntimeError(f'Only hashed binary wheels may be installed: {name}')
    return rows


def supplement(python, snapshot, packages, profile, cache, env):
    """Caller holds the exclusive lease; install only a reviewed additive plan."""
    if (snapshot['prefix'] == snapshot['base_prefix'] and
            not (Path(snapshot['prefix']) / 'conda-meta').is_dir()):
        raise RuntimeError('Will not install into system Python; select an existing Conda environment or venv')
    purelib = Path(snapshot['purelib'])
    if not purelib.is_dir() or not os.access(purelib, os.W_OK):
        raise RuntimeError(f'Existing environment is not writable: {purelib}')
    missing_requirements(snapshot, packages, profile)
    cache.mkdir(parents=True, exist_ok=True)
    from tempfile import mkdtemp
    attempt = Path(mkdtemp(prefix='supplement_', dir=cache))
    before = snapshot['packages']
    (attempt / 'before.json').write_text(json.dumps(snapshot, indent=2) + '\n')
    constraints = attempt / 'installed.constraints.txt'
    constraints.write_text(''.join(f'{name}=={version}\n' for name, version in sorted(before.items())))
    requirements = attempt / 'required.txt'
    requirements.write_text('\n'.join(plan_requirements(snapshot, packages, profile)) + '\n')
    install_env = installer_environment(env, cache, python)
    pip = pip_command(python, snapshot, cache)

    def run(*args, capture=False):
        result = subprocess.run([*pip, *args], env=install_env, text=True, capture_output=capture)
        if result.returncode and not capture:
            raise RuntimeError(f'Dependency preparation failed; see {attempt}. Installed versions were constrained; '
                               'no upgrade/downgrade was requested. Retry the same command after fixing connectivity.')
        return result

    baseline = run('check', capture=True)
    (attempt / 'pip_check_before.txt').write_text(baseline.stdout + baseline.stderr)
    report_file = attempt / 'plan.json'
    index = []
    if 'torch' in packages and 'torch' not in before:
        # Use the PyTorch index as the primary source so the CUDA/CPU torch stack
        # resolves from it, and keep the general index (cluster mirror if set,
        # else PyPI) as the fallback. This ordering matters: torch's pure-Python
        # deps (jinja2, markupsafe, ...) are published on the PyTorch index WITHOUT
        # a sha256, and pip does not prefer a hashed candidate over an equal-version
        # unhashed one. Making the general index the fallback lets those deps resolve
        # from a hashed wheel, which validate_plan() requires ("Only hashed binary
        # wheels may be installed").
        general = install_env.get('PIP_INDEX_URL', 'https://pypi.org/simple')
        index = ['--index-url', 'https://download.pytorch.org/whl/' +
                 ('cpu' if profile == 'cpu' else 'cu129'), '--extra-index-url', general]
    run('install', '--dry-run', '--report', str(report_file), '--only-binary=:all:',
        '-c', str(constraints), '-r', str(requirements), *index)
    rows = validate_plan(json.loads(report_file.read_text()), before)
    if not rows:
        raise RuntimeError('No missing dependencies can be added. Existing packages failed capability checks; '
                           f'they will not be reinstalled or replaced. Diagnostic: {attempt}')
    names = [f'{row["metadata"]["name"]}=={row["metadata"]["version"]}' for row in rows]
    print('INSTALLING MISSING DEPENDENCIES ONLY: ' + ', '.join(names), flush=True)
    wheelhouse = attempt / 'wheels'
    wheelhouse.mkdir()
    lock_lines = []
    from urllib.parse import urlparse, unquote
    for row in rows:
        url = row['download_info']['url']
        sha = row['download_info']['archive_info']['hashes']['sha256']
        wheel = wheelhouse / Path(unquote(urlparse(url).path)).name
        fetch_wheel(url, sha, wheel)
        lock_lines.append(f'{wheel.as_uri()} --hash=sha256:{sha}')
    lockfile = attempt / 'missing.lock'
    lockfile.write_text('\n'.join(lock_lines) + '\n')
    # Download and verify everything before the first environment mutation.
    # --no-deps + local hash lock makes a second dependency resolution impossible.
    run('install', '--no-index', '--no-deps', '--require-hashes', '-r', str(lockfile))
    after = inventory(python, env)
    changed = [n for n, v in before.items() if after['packages'].get(n) != v]
    if changed:
        raise RuntimeError(f'Installed packages unexpectedly changed: {changed}; stop and inspect {attempt}')
    final = run('check', capture=True)
    (attempt / 'pip_check_after.txt').write_text(final.stdout + final.stderr)
    old_errors = set((baseline.stdout + baseline.stderr).splitlines())
    new_errors = set((final.stdout + final.stderr).splitlines()) - old_errors
    if final.returncode and new_errors:
        raise RuntimeError(f'New dependency conflicts after supplement: {sorted(new_errors)}; see {attempt}')
    (attempt / 'after.json').write_text(json.dumps(after, indent=2) + '\n')
    return dict(packages=names, report=str(attempt))
