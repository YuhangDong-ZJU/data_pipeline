from __future__ import annotations

import argparse
from fractions import Fraction
from pathlib import Path
import tarfile
import tempfile
import unittest
import os

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from recam_refine.archives import unpack_archive, member_path
from recam_refine.common import RefineError, check_transform, read_json, read_jsonl, sha256, write_json, write_jsonl
from recam_refine.media import trim_video, assert_same_video_prefix, decode_check
from recam_refine.pipeline import run
from recam_refine.pointworld import release_pose
from recam_refine.stats import table_stats, aggregate


def make_video(path, n, shape=(48,64), fps=15):
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), 'w') as c:
        s = c.add_stream('libx264', rate=fps)
        s.width, s.height, s.pix_fmt = shape[1], shape[0], 'yuv420p'
        s.options = {'crf':'18', 'bf':'3', 'g':'30'}
        for i in range(n):
            a = np.random.default_rng(i).integers(0,256,(*shape,3),dtype=np.uint8)
            f = av.VideoFrame.from_ndarray(a, format='rgb24')
            f.pts, f.time_base = i, Fraction(1, fps)
            for p in s.encode(f):
                c.mux(p)
        for p in s.encode():
            c.mux(p)


def make_table(i, n, offset):
    k = np.repeat(np.array([[[50,0,32],[0,50,24],[0,0,1]]]*3, dtype=np.float32)[None], n, axis=0)
    e = np.tile(np.eye(4, dtype=np.float32), (n,3,1,1))
    return pa.table({
        'observation.states.joint_state': pa.array(np.zeros((n,7)).tolist(), type=pa.list_(pa.float32(),7)),
        'observation.states.gripper_state': pa.array(np.zeros((n,1)).tolist(), type=pa.list_(pa.float32(),1)),
        'observation.camera.intrinsics': pa.array(k.tolist(), type=pa.list_(pa.list_(pa.list_(pa.float32(),3),3),3)),
        'observation.camera.extrinsics': pa.array(e.tolist(), type=pa.list_(pa.list_(pa.list_(pa.float32(),4),4),3)),
        'timestamp':pa.array(np.arange(n)/15, type=pa.float32()),
        'frame_index':pa.array(np.arange(n),type=pa.int64()),
        'episode_index':pa.array([i]*n,type=pa.int64()),
        'index':pa.array(np.arange(offset,offset+n),type=pa.int64()),
        'task_index':pa.array([0]*n,type=pa.int64()),
    })


def fixture(root, droid=True):
    root.mkdir(parents=True)
    features = {}
    for cam in ((0,1,2) if droid else (0,)):
        for name, dtype, channels in [('rgb','video',3),('depth','image',1),('normal' if droid else 'normals','video',3)]:
            if droid and name == 'normal' and cam == 0:
                continue
            features[f'observation.images.{name}_{cam:02d}'] = dict(dtype=dtype,shape=[48,64,channels])
    lengths = [6,8] if droid else [4]
    offset, stats, episodes = 0, [], []
    for i,n in enumerate(lengths):
        t = make_table(i,n,offset)
        path = root / f'data/chunk-000/episode_{i:06d}.parquet'
        path.parent.mkdir(parents=True,exist_ok=True)
        pq.write_table(t,path)
        for field in t.schema:
            shape=[]
            ty=field.type
            while pa.types.is_fixed_size_list(ty):
                shape.append(ty.list_size)
                ty=ty.value_type
            features[field.name] = dict(dtype=str(ty),shape=shape or [1])
        for key, feature in features.items():
            if feature['dtype']=='video':
                count = 8 if droid and 'normal_' in key else n
                make_video(root/f'videos/chunk-000/{key}/episode_{i:06d}.mp4',count)
            elif feature['dtype']=='image':
                count = 8 if droid and 'depth_00' not in key else n
                for f in range(count):
                    p=root/f'images/chunk-000/{key}/episode_{i:06d}/frame_{f:06d}.png'
                    p.parent.mkdir(parents=True,exist_ok=True)
                    zero = droid and i==1 and f>=6 and key.endswith('01')
                    Image.fromarray(np.full((48,64),0 if zero else 1000,np.uint16)).save(p)
        episodes.append(dict(episode_index=i,length=n,tasks=['test task']))
        stats.append(dict(episode_index=i,stats=table_stats(t)))
        offset+=n
    info=dict(codebase_version='v2.1',features=features, total_episodes=len(lengths), total_frames=offset,
              total_tasks=1,total_videos=len(lengths)*sum(v['dtype']=='video' for v in features.values()),
              total_images=offset*sum(v['dtype']=='image' for v in features.values()),total_chunks=1,chunks_size=1000,
              fps=15,splits={'train':f'0:{len(lengths)}'},data_path='data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',
              video_path='videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4',
              image_path='images/chunk-{episode_chunk:03d}/{image_key}/episode_{episode_index:06d}/frame_{frame_index:06d}.png')
    write_json(root/'meta/info.json',info)
    write_jsonl(root/'meta/episodes.jsonl',episodes)
    write_jsonl(root/'meta/tasks.jsonl',[dict(task_index=0,task='test task')])
    write_jsonl(root/'meta/episodes_stats.jsonl',stats)
    write_json(root/'meta/stats.json',aggregate(stats))
    write_json(root/'meta/cameras.json',dict(cameras=[dict(camera_index=i) for i in range(3)],calibration={'extrinsics':{}}))
    write_json(root/'meta/coordinates.json',dict(robot_base_frame='panda_link0'))
    return info


class SafetyTests(unittest.TestCase):
    def test_transfer_replay_and_cross_filesystem(self):
        from recam_refine.pipeline import transfer_depth
        with tempfile.TemporaryDirectory() as src_tmp, tempfile.TemporaryDirectory(dir=os.environ.get('RECAM_TEST_SECOND_FILESYSTEM')) as dst_tmp:
            src, base = Path(src_tmp), Path(dst_tmp)
            root, work = base/'recam_lerobot', base/'work'
            droid=root/'real_world/droid'
            droid.mkdir(parents=True)
            row=dict(episode_index=0,length=2,source_episode_id='test+0',camera_serials={'external_1':'a','external_2':'b'})
            for cam,serial in ((1,'a'),(2,'b')):
                directory=src/f'images/chunk-000/observation.images.depth_{cam:02d}/episode_000000'
                directory.mkdir(parents=True)
                for frame in range(2):
                    Image.fromarray(np.full((720,1280),1000,np.uint16)).save(directory/f'frame_{frame:06d}.png')
                write_json(src/f'annotations/foundation_stereo_depth/chunk-000/observation.images.depth_{cam:02d}/episode_000000.json',
                    dict(source=dict(episode_index=0,source_episode_id='test+0',camera_role=f'external_{cam}',camera_serial=serial,
                        frame_count=2,decoded_frame_count=2,tail_missing_count=0,missing_frame_indices=[],timestamps_ms=[1.,2.]),
                         inference={'method':'FoundationStereo'},calibration={'intrinsic':np.eye(3).tolist()}))
            transfer_depth(root,droid,src,{0:row},{0},work)
            transfer_depth(root,droid,src,{0:row},{0},work)
            self.assertTrue(all(r['complete'] for r in (read_json(p) for p in (work/'transfer_receipts').glob('*.json'))))
            self.assertEqual(len(list(droid.glob('images/*/*/*/*.png'))),4)
            if src.stat().st_dev==droid.stat().st_dev:
                self.assertEqual(len(list(src.glob('images/*/*/episode_*'))),0)
            else:
                self.assertEqual(len(list(src.glob('images/*/*/*/*.png'))),4)

    def test_depth_objective_optimization(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest('GPU runtime not installed')
        from recam_refine.pointworld import refine_camera
        xy=np.stack(np.meshgrid(np.linspace(-.3,.3,20),np.linspace(-.2,.2,20)),-1).reshape(-1,2)
        points=np.column_stack([xy,np.ones(len(xy))]).astype(np.float32)
        depths=[np.ones((48,64),np.float32) for _ in range(6)]
        initial=np.eye(4)
        initial[2,3]=.04
        k=np.array([[50,0,32],[0,50,24],[0,0,1]],np.float32)
        result,metrics=refine_camera(initial,k,depths,[points]*6,device='cpu',iterations=160,min_points=10)
        self.assertTrue(metrics['accepted'])
        self.assertLess(metrics['final_holdout_loss'],.002)
        self.assertLess(abs(result[2,3]),.002)

    def test_tar_paths(self):
        archive='images/chunk-000/observation.images.depth_01/episodes-000000-000249.tar'
        for name in ('../escape.png','/etc/passwd','episode_000000/../../x','images/chunk-001/observation.images.depth_01/episode_000000/frame_000000.png'):
            with self.assertRaises(RefineError):
                member_path(name,archive)
        self.assertEqual(member_path('episode_000000/frame_000000.png',archive),
                         'images/chunk-000/observation.images.depth_01/episode_000000/frame_000000.png')

    def test_archive_conflict_keeps_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'dataset'
            path=root/'images/chunk-000/observation.images.depth_01/episode_000000/frame_000000.png'
            path.parent.mkdir(parents=True)
            Image.fromarray(np.ones((8,8),np.uint16)).save(path)
            archive=path.parent.parent/'episodes-000000-000000.tar'
            with tarfile.open(archive,'w') as tar:
                tar.add(path,arcname=path.relative_to(root).as_posix())
            unpack_archive(archive,root,Path(tmp)/'receipts')
            Image.fromarray(np.full((8,8),2,np.uint16)).save(path)
            digest=sha256(path)
            with self.assertRaises(RefineError):
                unpack_archive(archive,root,Path(tmp)/'receipts')
            self.assertEqual(sha256(path),digest)
            self.assertTrue(archive.exists())

    def test_b_frame_trim_has_exact_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/'a.mp4',Path(tmp)/'b.mp4'
            make_video(a,19)
            trim_video(a,b,17,15)
            decode_check(b,17,(48,64,3),15)
            assert_same_video_prefix(a,b,17)

    def test_release_inverse_and_serial(self):
        with tempfile.TemporaryDirectory() as tmp:
            pose=np.eye(4)
            pose[0,3]=.5
            row=dict(source_episode_id='lab+uuid',camera_serials={'external_1':'a','external_2':'b'})
            d=dict(uuid='lab+uuid',optimization_success=True,optimization_summary={'final_loss':.05},
                   a={'optimized_extrinsics':pose.tolist()},b={'optimized_extrinsics':pose.tolist()})
            write_json(Path(tmp)/'lab+uuid_cameras.json',d)
            result,status=release_pose(tmp,row)
            self.assertEqual(status,'pointworld_release')
            self.assertEqual(result[0,0,3],-.5)
            row['camera_serials']['external_1']='wrong'
            self.assertEqual(release_pose(tmp,row)[1],'serial_missing')

    def test_reject_reflection(self):
        pose=np.eye(4)
        pose[0,0]=-1
        with self.assertRaises(RefineError):
            check_transform(pose,'test')


class EndToEndTests(unittest.TestCase):
    def test_mixed_previous_trim_tar_normals_metadata_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp)
            root=tmp/'recam_lerobot'
            droid=root/'real_world/droid'
            simulation=root/'simulation/test_sim'
            fixture(droid)
            fixture(simulation,False)
            tar_paths=[]
            for subset,key in ((droid,'01'),(simulation,'00')):
                camera=subset/f'images/chunk-000/observation.images.depth_{key}'
                archive=camera/'episodes-000000-000001.tar'
                with tarfile.open(archive,'w') as tar:
                    for p in sorted(camera.glob('episode_*/*.png')):
                        tar.add(p,arcname=p.relative_to(subset).as_posix())
                for p in list(camera.glob('episode_*/*.png')):
                    p.unlink()
                tar_paths.append(archive)
            manifest=tmp/'manifest.jsonl'
            write_jsonl(manifest,[dict(episode_index=i,length=8,source_episode_id=f'lab+{i}',
                          tasks=['test task'],camera_serials={'external_1':'a','external_2':'b'}) for i in range(2)])
            cameras=tmp/'cameras'
            c2w=np.eye(4)
            c2w[0,3]=.02
            for i in range(2):
                write_json(cameras/f'lab+{i}_cameras.json',dict(uuid=f'lab+{i}',optimization_success=True,
                    optimization_summary={'final_loss':.04},a={'optimized_extrinsics':np.linalg.inv(c2w).tolist()},
                    b={'optimized_extrinsics':np.linalg.inv(c2w).tolist()}))
            (droid/'logs').mkdir()
            (droid/'logs/run.log').write_text('keep this log')
            args=argparse.Namespace(root=root,work_dir=tmp/'work',depth_output=None,depth_chunks='2-13',depth_metadata=[],
                episode_manifest=manifest,pointworld_cameras=cameras,workers=2,devices='cpu',iterations=1)
            args.defer_cleanup=True
            run(args)
            self.assertTrue(tar_paths[0].exists())
            self.assertTrue((droid/'logs/run.log').exists())
            self.assertFalse((tmp/'work/SUCCESS.json').exists())
            self.assertTrue((tmp/'work/READY_FOR_GEOMETRY_AUDIT.json').exists())
            args.defer_cleanup=False
            run(args)
            info=read_json(droid/'meta/info.json')
            self.assertEqual(info['total_frames'],12)
            self.assertEqual(info['total_videos'],10)
            self.assertEqual(info['total_images'],24)
            self.assertNotIn('observation.images.depth_00',info['features'])
            self.assertFalse(tar_paths[0].exists())
            self.assertTrue(tar_paths[1].exists())
            self.assertFalse((droid/'logs/run.log').exists())
            self.assertTrue((tmp/'work/auxiliary/real_world/droid/logs/run.log').exists())
            self.assertEqual([r['length'] for r in read_jsonl(droid/'meta/episodes.jsonl')],[6,6])
            self.assertTrue((tmp/'work/SUCCESS.json').exists())
            before={str(p.relative_to(root)):sha256(p) for p in root.rglob('*') if p.is_file()}
            run(args)
            after={str(p.relative_to(root)):sha256(p) for p in root.rglob('*') if p.is_file()}
            self.assertEqual(before,after)
            # Replay apply after a simulated interruption before metadata commit.
            (tmp/'work/SUCCESS.json').unlink()
            for stage in ('05_apply','06_check','07_cleanup'):
                (tmp/'work'/f'{stage}.complete.json').unlink()
            (tmp/'work/applied/episode_000001.json').unlink()
            run(args)
            self.assertEqual(before,{str(p.relative_to(root)):sha256(p) for p in root.rglob('*') if p.is_file()})


if __name__=='__main__':
    unittest.main()
