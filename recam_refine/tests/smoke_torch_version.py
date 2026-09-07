"""An existing GPU Torch version is rejected before any package installation."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime-work-dir',type=Path,required=True)
    args = parser.parse_args()
    base = (args.runtime_work_dir/'runtime').resolve()
    repo = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix='recam_torch_version_') as td:
        root = Path(td)/'runtime'
        root.mkdir()
        (root/'uv').symlink_to(base/'uv')
        subprocess.run([str(base/'uv'),'venv','--python',str(base/'env/bin/python'),str(root/'env')],check=True)
        metadata = root/'env/lib/python3.11/site-packages/torch-2.5.1+cu124.dist-info/METADATA'
        metadata.parent.mkdir()
        original = 'Metadata-Version: 2.1\nName: torch\nVersion: 2.5.1+cu124\n'
        metadata.write_text(original)
        env = dict(os.environ,HTTP_PROXY='http://127.0.0.1:1',HTTPS_PROXY='http://127.0.0.1:1')
        result = subprocess.run(['python3','recam_refine/bootstrap.py',td,'--prepare-gpu'],cwd=repo,env=env,
                                text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=30)
        assert result.returncode!=0,result.stdout
        assert 'Existing runtime has torch 2.5.1+cu124; requested 2.8.0+cu129' in result.stdout,result.stdout
        assert 'prepare a NEW runtime work directory' in result.stdout,result.stdout
        assert metadata.read_text()==original
        assert not (metadata.parent.parent/'torch').exists()
        assert not (root/'cache').exists()
    print('GPU Torch version guard passed: old metadata unchanged; no package download or installation.')


if __name__=='__main__':
    main()
