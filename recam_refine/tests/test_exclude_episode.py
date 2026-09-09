import copy
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from recam_refine.common import write_json,write_jsonl,read_json,read_jsonl,values
from recam_refine.exclude_episode import execute
from recam_refine.inputs import canonical_manifest,load_depth_records
from recam_refine.scan_depth_timestamps import inspect
from recam_refine.stats import table_stats,aggregate


class ExclusionTest(unittest.TestCase):
    def test_transfer_uses_source_chunk_and_ignores_excluded_sidecar(self):
        from recam_refine.tests.test_steps import source_depth
        from recam_refine.pipeline import transfer_depth
        from recam_refine.steps import transfer_configuration
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);root=base/'dataset';droid=root/'real_world/droid'
            droid.mkdir(parents=True);source=base/'source';work=base/'work'
            row=dict(episode_index=1000,length=2,source_episode_id='replacement')
            source_depth(source,[row])
            # Bad original episode 0 is absent from the source identity map.
            write_json(source/'annotations/foundation_stereo_depth/chunk-000/observation.images.depth_01/episode_000000.json',
                       {'source':{'episode_index':0}})
            manifest={0:dict(row,episode_index=0,source_episode_index=1000,
                             camera_serials={'external_1':'a','external_2':'b'})}
            self.assertEqual(len(transfer_configuration(root,source,{0},manifest)['identities']),0)
            self.assertEqual(len(transfer_configuration(root,source,{1},manifest)['identities']),1)
            transfer_depth(root,droid,source,manifest,{1},work)
            self.assertTrue((droid/'images/chunk-000/observation.images.depth_01/episode_000000/frame_000000.png').is_file())

    def test_quarantine_tar_resume_indices_and_source_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp); root=base/'dataset'; work=base/'work'
            d=root/'real_world/droid'; (d/'meta').mkdir(parents=True)
            info=dict(codebase_version='v2.1',chunks_size=1000,total_episodes=6797,
                      total_frames=7143,total_tasks=1,total_chunks=7,splits={'train':'0:6797'},
                      features={},data_path='data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet')
            episodes=[]; stats=[]; offset=0
            for i in range(6797):
                n=345 if i==6795 else 3 if i==6796 else 1
                t=pa.table({'episode_index':np.full(n,i),'index':np.arange(offset,offset+n),
                            'frame_index':np.arange(n),'action':np.full(n,float(i))})
                p=d/f'data/chunk-{i//1000:03d}/episode_{i:06d}.parquet'
                p.parent.mkdir(parents=True,exist_ok=True); pq.write_table(t,p)
                episodes.append(dict(episode_index=i,length=n,tasks=['task']))
                stats.append(dict(episode_index=i,stats=table_stats(t))); offset+=n
            info['total_frames']=offset
            write_json(d/'meta/info.json',info);write_jsonl(d/'meta/episodes.jsonl',episodes)
            write_jsonl(d/'meta/episodes_stats.jsonl',stats);write_json(d/'meta/stats.json',aggregate(stats))
            write_jsonl(d/'meta/tasks.jsonl',[dict(task_index=0,task='task')])
            src=base/'source.json'
            side=dict(source=dict(episode_index=6795,source_episode_id='IPRL+w026bb9b+2023-09-22-19h-01m-34s',
                camera_role='external_1',camera_serial='27432424',frame_count=345,decoded_frame_count=345,
                initial_decoded_frame_count=344,tail_retry_attempted=True,tail_retry_recovered_frames=1,
                tail_missing_count=0,missing_frame_indices=[],timestamps_ms=list(range(344))+[343]))
            write_json(src,side)
            report=base/'scan.json';write_json(report,dict(affected_episode_ids=['006795'],
                duplicate_final_retry_episode_ids=['006795'],anomalies=[inspect(src)]))
            # Shared TAR contains a normal episode, the bad episode and the replacement.
            archive=d/'images/chunk-006/observation.images.depth_01/episodes-006794-006796.tar'
            archive.parent.mkdir(parents=True,exist_ok=True)
            png=io.BytesIO();Image.fromarray(np.full((2,2),42,dtype=np.uint16)).save(png,format='PNG'); blob=png.getvalue()
            with tarfile.open(archive,'w') as tar:
                for i in (6794,6795,6796):
                    m=tarfile.TarInfo(f'images/chunk-006/observation.images.depth_01/episode_{i:06d}/frame_000000.png')
                    m.size=len(blob);tar.addfile(m,io.BytesIO(blob))
            for i in (6795,6796):
                p=d/f'videos/chunk-006/observation.images.rgb_00/episode_{i:06d}.mp4'
                p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(str(i).encode())
            import recam_refine.exclude_episode as module
            original_copy=module.shutil.copy2
            def fail_install(src,dst,*a,**kw):
                if str(dst).endswith('.exclude-part'):
                    raise RuntimeError('injected interruption')
                return original_copy(src,dst,*a,**kw)
            with patch.object(module.shutil,'copy2',fail_install):
                with self.assertRaisesRegex(RuntimeError,'injected'):
                    execute(root,work,report)
            execute(root,work,report)
            execute(root,work,report)
            out=read_jsonl(d/'meta/episodes.jsonl')
            self.assertEqual(len(out),6796);self.assertEqual(out[6795]['source_episode_index'],6796)
            self.assertEqual((d/'videos/chunk-006/observation.images.rgb_00/episode_006795.mp4').read_bytes(),b'6796')
            self.assertFalse((d/'data/chunk-006/episode_006796.parquet').exists())
            table=pq.read_table(d/'data/chunk-006/episode_006795.parquet')
            self.assertEqual(values(table['index']).ravel().tolist(),[6795,6796,6797])
            self.assertTrue(np.all(values(table['action'])==6796))
            self.assertEqual(read_json(d/'meta/info.json')['total_frames'],6798)
            with tarfile.open(archive) as tar:
                self.assertEqual(len(tar.getnames()),1);self.assertIn('006794',tar.getnames()[0])
            self.assertEqual((d/'images/chunk-006/observation.images.depth_01/episode_006795/frame_000000.png').read_bytes(),blob)
            manifest=base/'manifest.jsonl'
            write_jsonl(manifest,[dict(episode_index=6796,source_episode_id='good',length=3,tasks=['task'],
                camera_serials={'external_1':'a','external_2':'b'})])
            mapped=canonical_manifest(manifest,[out[6795]])
            self.assertEqual(mapped[6795]['source_episode_index'],6796)
            depth=base/'annotations/chunk-006/observation.images.depth_01/episode_006796.json'
            write_json(depth,dict(source=dict(episode_index=6796,source_episode_id='good',camera_role='external_1',
                camera_serial='a',frame_count=3,decoded_frame_count=3,tail_missing_count=0,
                missing_frame_indices=[],timestamps_ms=[1,2,3]),inference={'method':'FoundationStereo'},calibration={'intrinsic':[]}))
            self.assertIn((6795,1),load_depth_records([base/'annotations'],mapped))


if __name__=='__main__':
    unittest.main()
