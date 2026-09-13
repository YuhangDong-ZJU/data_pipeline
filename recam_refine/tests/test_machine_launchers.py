"""Linux entrypoint orchestration tests with disposable, instrumented stage commands."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


@unittest.skipUnless(os.name == 'posix', 'Linux Bash')
class MachineLaunchers(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='recam_launchers_')
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.repo = self.base/'repo'
        source = Path(__file__).resolve().parents[2]
        shutil.copytree(source/'recam_refine/launchers',self.repo/'recam_refine/launchers')
        for name in ('prepare','gpu1','gpu2','cpu_finish'):
            shutil.copy2(source/f'run_{name}.sh',self.repo)
        (self.repo/'recam_refine/__init__.py').touch()
        self.work = self.base/'work'
        self.root = self.base/'dataset'
        (self.root/'real_world/droid').mkdir(parents=True)
        self.env = {k:v for k,v in os.environ.items() if not k.startswith('RECAM_')}
        self.env.update(RECAM_ROOT=str(self.root), RECAM_WORK=str(self.work),
                        DEPTH_OUTPUT=str(self.base/'depth'), WORKER_A=str(self.base/'worker0'),
                        WORKER_B=str(self.base/'worker1'), PATH=str(Path(sys.executable).parent)+os.pathsep+os.environ['PATH'])
        self.work.mkdir()
        report = dict(root=str(self.base/'depth/annotations/foundation_stereo_depth'),
                      by_chunk={f'chunk-{i:03d}':{} for i in range(2,14)},
                      affected_episode_ids=['006795'], duplicate_final_retry_episode_ids=['006795'])
        (self.work/'depth_timestamp_scan_existing.json').write_text(json.dumps(report))
        (self.repo/'recam_refine/environment.py').write_text('''
import json,sys
from pathlib import Path
work=Path(sys.argv[1]); profile=sys.argv[sys.argv.index('--profile')+1]
with (work/'trace.jsonl').open('a') as f: f.write(json.dumps(['environment',profile])+'\\n')
cache=work/'runtime_cache'; cache.mkdir(exist_ok=True)
(cache/f'environment_{profile}.json').write_text(json.dumps({'python':sys.executable}))
''')
        (self.repo/'recam_refine/run_step.sh').write_text('''#!/usr/bin/env bash
set -euo pipefail
exec python3 recam_refine/mock_step.py "$@"
''')
        (self.repo/'recam_refine/mock_step.py').write_text('''
import hashlib,json,os,sys
from pathlib import Path
stage,root,work=sys.argv[1:4]; work=Path(work)
with (work/'trace.jsonl').open('a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')
assert '--no-install' in sys.argv and '--python' in sys.argv
if os.environ.get('FAIL_STEP')==stage: raise SystemExit(17)
if stage=='shard-refine':
    shard=sys.argv[sys.argv.index('--shard-id')+1]
    (work/f'gpu{shard}.done').touch()
    plan=json.loads((work/'shards/plan.json').read_text()); part=plan['shards'][int(shard)]
    directory=work/f'shards/results/shard-{int(shard):05d}'; (directory/'cameras').mkdir(parents=True,exist_ok=True)
    (directory/'COMPLETE.json').write_text(json.dumps({'complete':True,'plan_id':plan['plan_id'],
      'shard_id':int(shard),'manifest_sha256':part['sha256'],'candidate_sha256':{}}))
    raise SystemExit(0)
if stage=='shard-merge' and not all((work/f'gpu{i}.done').exists() for i in (0,1)):
    raise SystemExit(19)
markers={'exclude-6795':'exclude_episode_006795/SUCCESS.json','transfer':'STEP1_DEPTH_TRANSFER_SUCCESS.json',
'unpack':'STEP2_UNPACK_SUCCESS.json','align':'STEP3_ALIGN_SUCCESS.json','overlap':'STEP4_OVERLAP_SUCCESS.json',
'shard-plan':'SHARD_PLAN_READY.json','shard-merge':'STEP4_REFINE_SUCCESS.json','apply':'STEP4_APPLY_SUCCESS.json',
'check':'STEP5_CHECK_SUCCESS.json','cleanup':'SUCCESS.json','repack':'REPACK_SUCCESS.json'}
p=work/markers[stage]; p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps({'root':root,'complete':True}))
if stage=='shard-plan':
    plan=work/'shards/plan.json'; plan.parent.mkdir(exist_ok=True)
    parts=[]
    for i in (0,1):
        manifest=work/f'shards/manifest{i}.json'; manifest.write_text('{}')
        parts.append({'shard_id':i,'path':str(manifest.relative_to(work)),
                      'sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'episodes':[]})
    plan.write_text(json.dumps({'plan_id':'test','shards':parts,'settings':{'backend':'batched'}}))
    p.write_text(json.dumps({'plan_sha256':hashlib.sha256(plan.read_bytes()).hexdigest(),'complete':True}))
''')

        # A local bare origin exercises real fetch/pull without network access.
        self.origin = self.base/'origin.git'
        self.git('init','--bare',str(self.origin),cwd=self.base)
        self.git('init','-b','codex/recam-dataset-refine')
        self.git('config','user.email','test@example.invalid')
        self.git('config','user.name','Launcher Test')
        self.git('add','.')
        self.git('commit','-m','initial fixture')
        self.git('remote','add','origin',str(self.origin))
        self.git('push','-u','origin','codex/recam-dataset-refine')

    def git(self,*args,cwd=None):
        return subprocess.run(['git',*args],cwd=cwd or self.repo,check=True,
                              capture_output=True,text=True).stdout

    def launch(self, name, *args, success=True, **extra):
        result = subprocess.run(['bash',str(self.repo/f'run_{name}.sh'),*args],
                                env={**self.env,**extra},capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode==0,success,result.stdout+result.stderr)
        return result

    def trace(self):
        return [json.loads(line) for line in (self.work/'trace.jsonl').read_text().splitlines()]

    def test_full_sequence_missing_shard_and_resume(self):
        self.launch('prepare')
        self.assertEqual([r[0] for r in self.trace()],['environment','environment','exclude-6795',
                         'transfer','unpack','align','overlap','shard-plan'])
        saved = self.trace()
        result=self.launch('prepare')
        self.assertIn('已完成，因此跳过',result.stdout)
        self.assertEqual(self.trace(),saved)
        self.launch('gpu1')
        self.launch('cpu_finish',success=False)
        self.assertNotIn('apply',[r[0] for r in self.trace()])
        self.launch('gpu2')
        self.launch('cpu_finish')
        stages = [r[0] for r in self.trace()]
        self.assertEqual(stages[-5:],['shard-merge','apply','check','cleanup','repack'])
        count=len(stages)
        self.launch('cpu_finish')
        self.assertEqual(len(self.trace()),count)
        for machine in ('gpu1','gpu2'):
            result=self.launch(machine)
            self.assertIn('已完成，因此跳过',result.stdout)
        self.assertEqual(len(self.trace()),count)
        self.assertEqual(stages.count('check'),1)
        self.assertEqual(stages.count('environment'),2)
        candidate=self.work/'shards/results/shard-00000/cameras/episode_999999.json'
        candidate.write_text('{}')
        result=self.launch('gpu1',success=False)
        self.assertIn('coverage changed',result.stderr+result.stdout)
        self.assertEqual(len(self.trace()),count)

    def test_failed_preparation_resumes_without_repeating_mutations(self):
        self.launch('prepare',success=False,FAIL_STEP='align')
        self.assertFalse((self.work/'launchers/PREPARE_READY.json').exists())
        self.assertNotIn('overlap',[r[0] for r in self.trace()])
        self.launch('prepare')
        stages=[r[0] for r in self.trace()]
        for name in ('exclude-6795','transfer','unpack'):
            self.assertEqual(stages.count(name),1)
        self.assertEqual(stages.count('align'),2)

    def test_dry_run_and_frozen_configuration(self):
        self.launch('prepare','--dry-run')
        self.assertFalse((self.work/'trace.jsonl').exists())
        self.assertFalse((self.work/'launchers').exists())
        self.launch('prepare')
        self.launch('gpu1',success=False,AUDIT_FRAMES='12')
        self.assertNotIn('shard-refine',[r[0] for r in self.trace()])
        with (self.repo/'recam_refine/mock_step.py').open('a') as stream:
            stream.write('\n# code changed\n')
        self.launch('gpu1')  # Compatible code updates do not rerun CPU preparation.

    def test_newest_wrong_report_blocks_exclusion(self):
        report = self.work/'depth_timestamp_scan_new.json'
        report.write_text('{}')
        os.utime(report,(2000000000,2000000000))
        self.launch('prepare',success=False)
        self.assertNotIn('exclude-6795',[r[0] for r in self.trace()])

    def test_old_workflow_lock_is_ignored(self):
        import fcntl
        self.launch('prepare')
        with (self.work/'launchers/workflow.lock').open('a') as lease:
            fcntl.flock(lease,fcntl.LOCK_SH|fcntl.LOCK_NB)
            self.launch('gpu1')
            self.launch('gpu2')
            self.launch('prepare')
            self.launch('cpu_finish')
        self.launch('cpu_finish')

    def test_pull_new_entry_then_skip_completed_data_steps(self):
        self.launch('prepare')
        previous = [r[0] for r in self.trace() if r[0]!='environment']
        publisher=self.base/'publisher'
        self.git('clone',str(self.origin),str(publisher),cwd=self.base)
        self.git('switch','codex/recam-dataset-refine',cwd=publisher)
        self.git('config','user.email','test@example.invalid',cwd=publisher)
        self.git('config','user.name','Launcher Test',cwd=publisher)
        entry=publisher/'run_prepare.sh'
        entry.write_text(entry.read_text().replace('set -Eeuo pipefail','set -Eeuo pipefail\necho NEW_ENTRY_EXECUTED'))
        self.git('add','run_prepare.sh',cwd=publisher)
        self.git('commit','-m','update entry',cwd=publisher)
        self.git('push','origin','codex/recam-dataset-refine',cwd=publisher)
        result=self.launch('prepare')
        self.assertIn('NEW_ENTRY_EXECUTED',result.stdout)
        self.assertIn('已完成，因此跳过',result.stdout)
        self.assertEqual(previous,[r[0] for r in self.trace() if r[0]!='environment'])
        self.assertTrue((self.work/'launchers/code_updates.jsonl').is_file())
        self.launch('gpu1')

    def test_local_tracked_changes_stop_update_before_processing(self):
        entry=self.repo/'run_gpu1.sh'
        entry.write_text(entry.read_text()+'\n# local edit\n')
        result=self.launch('prepare',success=False)
        self.assertIn('tracked local changes',result.stdout+result.stderr)
        self.assertFalse((self.work/'trace.jsonl').exists())
        self.assertIn('# local edit',entry.read_text())


if __name__ == '__main__':
    unittest.main()
