"""Exercise delivered Bash/CLI commands on a disposable synthetic dataset."""
import argparse
from pathlib import Path
import subprocess
import tempfile

from recam_refine.common import read_json
from recam_refine.steps import MARKERS
from recam_refine.tests.test_steps import all_files, manual_fixture


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime-work-dir',type=Path,required=True)
    parser.add_argument('--gpu-bootstrap',action='store_true')
    args = parser.parse_args()
    runtime = (args.runtime_work_dir/'runtime').resolve()
    assert (runtime/'env/bin/python').is_file()
    repo = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix='recam_manual_cli_') as tmp:
        fixture,droid,archives = manual_fixture(Path(tmp))
        fixture.work_dir.mkdir()
        # Reuse the independently bootstrapped CI/test runtime, while keeping
        # every workflow marker, dataset and source in the disposable fixture.
        (fixture.work_dir/'runtime').symlink_to(runtime,target_is_directory=True)
        def shell(step,*extra,success=True):
            result = subprocess.run(['bash','recam_refine/run_step.sh',step,str(fixture.root),str(fixture.work_dir),*extra],cwd=repo)
            assert (result.returncode==0)==success,(step,result.returncode)
        shell('transfer','--depth-output',str(fixture.depth_output),'--depth-chunks','0',
              '--episode-manifest',str(fixture.episode_manifest))
        assert (fixture.work_dir/MARKERS['transfer']).exists()
        assert not (fixture.work_dir/MARKERS['unpack']).exists()
        assert read_json(droid/'meta/info.json')['total_frames']==14
        shell('unpack')
        assert all(p.exists() for p in archives)
        shell('align','--episode-manifest',str(fixture.episode_manifest),'--workers','2')
        assert read_json(droid/'meta/info.json')['total_frames']==12
        before = all_files(droid)
        shell('overlap','--pointworld-cameras',str(fixture.pointworld_cameras))
        if args.gpu_bootstrap:
            shell('refine')
        else:
            # All fixture cameras have a release: CLI needs no Torch/GPU here.
            subprocess.run([str(runtime/'env/bin/python'),'-m','recam_refine','run-step','refine',str(fixture.root),
                            '--work-dir',str(fixture.work_dir),'--devices','cpu'],cwd=repo,check=True)
        assert all_files(droid)==before
        shell('apply','--workers','2')
        shell('status')
        shell('cleanup',success=False)
        assert all(p.exists() for p in archives)
        assert not (fixture.work_dir/'SUCCESS.json').exists()
        print('Manual Bash/CLI smoke passed: independent stages stop, candidates do not write back, unchecked cleanup rejected.',flush=True)


if __name__=='__main__':
    main()
