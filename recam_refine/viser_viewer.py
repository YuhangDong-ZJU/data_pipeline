"""Optional local Viser viewer. Reads portable bundles; never opens a dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import threading

import numpy as np
from PIL import Image


COLORS = ((237,96,85),(23,166,198))


def load_bundle(directory):
    metadata = json.loads((directory/'scene.json').read_text(encoding='utf-8'))
    for name, expected in metadata['artifact_sha256'].items():
        path = (directory/name).resolve()
        if not path.is_relative_to(directory.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError(f'Viewer bundle changed: {name}')
    with np.load(directory/'scene.npz',allow_pickle=False) as data:
        arrays = {k:data[k] for k in data.files}
    images = [np.array(Image.open(directory/f'source_camera{c}.png').convert('RGB')) for c in (1,2)]
    return metadata, arrays, images


def serve(directory, port, export_dir=None):
    import viser
    import viser.transforms as tf

    metadata, data, images = load_bundle(directory)
    server = viser.ViserServer(host='127.0.0.1',port=port,verbose=False)
    server.scene.set_up_direction('+z')
    server.scene.world_axes.visible = False
    server.scene.configure_default_lights(cast_shadow=False)
    server.gui.configure_theme(control_width='medium',dark_mode=False,
                               show_share_button=False,brand_color=(35,119,128))
    server.gui.main_panel.dock_right()
    server.gui.set_panel_label('ReCam · 点云与机器人对齐')
    position, target = (1.25,-1.35,.9), (.32,0,.22)
    server.initial_camera.position = position
    server.initial_camera.look_at = target
    server.initial_camera.up_direction = (0,0,1)
    server.initial_camera.fov = np.deg2rad(48)
    server.initial_camera.near = .005

    groups, clouds, cameras = {}, {}, {}
    for variant in ('before','after'):
        groups[variant] = server.scene.add_frame('/'+variant,show_axes=False,visible=variant=='after')
        for i in range(2):
            prefix = f'/{variant}/camera{i+1}'
            clouds[variant,i] = server.scene.add_point_cloud(prefix+'/points',
                points=data[f'camera{i+1}_{variant}'],colors=data[f'camera{i+1}_rgb'].copy(),
                point_size=.002,point_shape='circle',point_shading='flat',precision='float32')
            pose = np.array(metadata['poses'][variant][i])
            k = np.array(metadata['intrinsics'][i])
            h,w = images[i].shape[:2]
            cameras[variant,i] = server.scene.add_camera_frustum(prefix+'/frustum',
                fov=2*np.arctan(h/(2*k[1,1])),aspect=(w/h)*(k[1,1]/k[0,0]),
                image=images[i],color=COLORS[i],scale=.12,thickness=.003,
                wxyz=tf.SO3.from_matrix(pose[:3,:3]).wxyz,position=pose[:3,3],cast_shadow=False)
    mesh = server.scene.add_mesh_simple('/robot_reference',data['robot_vertices'],data['robot_faces'],
        color=(205,210,218),opacity=.28,side='double',cast_shadow=False,receive_shadow=False)
    axes = server.scene.add_frame('/base_axes',axes_length=.12,axes_radius=.002,visible=False)

    server.gui.add_markdown(f'**Episode {metadata["episode_index"]} · 第 {metadata["frame_index"]} 帧**\n\n'
        '拖动旋转，滚轮缩放。切换前后时观察角度保持不变。')
    mode = server.gui.add_dropdown('外参',options=('优化前','优化后'),initial_value='优化后')
    status = server.gui.add_markdown('**当前：'+metadata['after_label']+'**')
    coloring = server.gui.add_dropdown('点云颜色',options=('真实 RGB','按相机着色'),initial_value='真实 RGB')
    point_size = server.gui.add_slider('点大小 / mm',min=.5,max=6.,step=.5,initial_value=2.)
    opacity = server.gui.add_slider('模型不透明度',min=0.,max=1.,step=.05,initial_value=.3,
        hint='0 为完全透明，1 为不透明。模型由该帧关节状态和 URDF 确定。')
    mesh.opacity = opacity.value
    show_robot = server.gui.add_checkbox('显示机器人参考模型',initial_value=True)
    show_cameras = server.gui.add_checkbox('显示相机与 RGB',initial_value=True)
    show_axes = server.gui.add_checkbox('显示基座坐标轴',initial_value=False)
    cam_checks = [server.gui.add_checkbox(f'显示相机 {i+1} 点云',initial_value=True) for i in range(2)]
    reset = server.gui.add_button('恢复初始视角')
    server.gui.add_markdown('灰色模型是运动学参照；只切换外参，不改变深度。\n\n'
        '双色模式：红色 = 相机 1，青色 = 相机 2。')

    def update(_=None):
        variant = 'before' if mode.value=='优化前' else 'after'
        with server.atomic():
            for key, group in groups.items():
                group.visible = key==variant
            for (key,i), cloud in clouds.items():
                cloud.colors = (data[f'camera{i+1}_rgb'] if coloring.value=='真实 RGB'
                                else np.tile(np.array(COLORS[i],dtype=np.uint8),(len(cloud.points),1)))
                cloud.point_size = point_size.value/1000
                cloud.visible = cam_checks[i].value
                cameras[key,i].visible = show_cameras.value
            mesh.visible = show_robot.value
            mesh.opacity = opacity.value
            axes.visible = show_axes.value
            status.content = '**当前：'+metadata[variant+'_label']+'**'
    for control in [mode,coloring,point_size,opacity,show_robot,show_cameras,show_axes,*cam_checks]:
        control.on_update(update)

    @reset.on_click
    def reset_camera(event):
        if event.client is not None:
            event.client.camera.position = position
            event.client.camera.look_at = target
            event.client.camera.up_direction = (0,0,1)

    if export_dir:
        export_dir.mkdir(parents=True,exist_ok=True)
        for variant in ('before','after'):
            for key,group in groups.items():
                group.visible = key==variant
            server.flush()
            (export_dir/f'{variant}.html').write_text(server.get_scene_serializer().as_html(),encoding='utf-8')
        update()
    print(f'RECAM VISER http://127.0.0.1:{port}',flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        server.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle',type=Path,help='Directory containing scene.json and scene.npz')
    parser.add_argument('--port',type=int,default=8870)
    parser.add_argument('--export-dir',type=Path,help='Also export standalone before/after HTML snapshots')
    args = parser.parse_args()
    serve(args.bundle.resolve(),args.port,args.export_dir)


if __name__ == '__main__':
    main()
