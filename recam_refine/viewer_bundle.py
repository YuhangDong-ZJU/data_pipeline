"""Prepare an audited, portable scene for the optional Viser viewer."""
from pathlib import Path
import shutil

import numpy as np
import pyarrow.parquet as pq

from .common import require, read_json, write_json, sha256, parquet_path, values
from .pointworld import Robot


def prepare_viewer(args):
    root, output = args.root.resolve(), args.output_dir.resolve()
    require(not output.is_relative_to(root), 'Viewer output must be outside the dataset')
    if (root/'real_world/droid').is_dir():
        root = root/'real_world/droid'
    fusion = args.fusion_dir.resolve()
    require(output != fusion, 'Use a separate directory; preserve the fusion report')
    metadata = read_json(fusion/'provenance.json')
    report = read_json(args.metrics)
    require(metadata['source_sha256'].get(str(args.metrics.resolve())) == sha256(args.metrics),
            'Fusion preview does not match this audit report')
    sources = {str(args.metrics.resolve()):sha256(args.metrics)}
    for name in ('paired_clouds.npz','source_camera1.png','source_camera2.png'):
        path = fusion/name
        sources[str(path)] = sha256(path)
        require(sources[str(path)] == metadata['artifact_sha256'][name], f'Fusion artifact changed: {name}')
    episode, frame = metadata['episode_index'], metadata['frame_index']
    info = read_json(root/'meta/info.json')
    parquet = parquet_path(root,info,episode)
    sources[str(parquet)] = sha256(parquet)
    require(sources[str(parquet)] == report['input_sha256'].get(str(parquet)), 'Parquet changed since audit')
    table = pq.read_table(parquet,columns=['observation.states.joint_state','observation.states.gripper_state'])
    joints = values(table['observation.states.joint_state'])[frame]
    gripper = float(values(table['observation.states.gripper_state'])[frame].item())
    robot = Robot(args.robot_urdf,keep_meshes=True)
    vertices, faces = robot.geometry(joints,gripper)
    with np.load(fusion/'paired_clouds.npz',allow_pickle=False) as data:
        arrays = {k:data[k] for k in data.files}
    arrays.update(robot_vertices=vertices.astype(np.float32),robot_faces=faces.astype(np.uint32))
    output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/'scene.npz',**arrays)
    for c in (1,2):
        shutil.copyfile(fusion/f'source_camera{c}.png',output/f'source_camera{c}.png')
    require(all(sha256(Path(p))==h for p,h in sources.items()), 'Input changed while preparing viewer')
    write_json(output/'scene.json',dict(episode_index=episode,frame_index=frame,
        before_label='DROID 初值',after_label='PointWorld 发布外参' if metadata['after']=='pointworld_release' else 'PointWorld 方法优化',
        poses={'before':report['poses']['droid_initial'],'after':report['poses'][metadata['after']]},
        intrinsics=report['intrinsics_half_resolution'],joints=joints.tolist(),gripper=gripper,
        robot_urdf=str(args.robot_urdf.resolve()),robot_urdf_sha256=sha256(args.robot_urdf),
        source_sha256=sources,inputs_unchanged=True,
        artifact_sha256={p.name:sha256(p) for p in output.iterdir() if p.suffix in ('.npz','.png')}))
    print(f'VIEWER BUNDLE {output}',flush=True)
