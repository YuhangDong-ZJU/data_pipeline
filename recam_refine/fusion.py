"""Read-only, synchronized two-view fusion previews from an existing audit."""
from __future__ import annotations

import html
import os
from pathlib import Path

import numpy as np
from PIL import Image

from .audit import read_rgb
from .common import require, read_json, write_json, media_path, sha256, check_transform
from .geometry_metrics import WORKSPACE_MIN, WORKSPACE_MAX


VIEW_COLORS = ("#ed6055", "#17a6c6")


def paired_cloud(depth, rgb, intrinsic, before, after, bounds=(WORKSPACE_MIN, WORKSPACE_MAX)):
    """Use identical source pixels for both poses, with a union workspace crop."""
    require(depth.shape == rgb.shape[:2], "RGB/depth shapes differ")
    valid = np.isfinite(depth) & (depth > 0) & (depth <= 4)
    y, x = np.nonzero(valid)
    rays = np.column_stack([x, y, np.ones(len(x))]) @ np.linalg.inv(intrinsic).T
    xyz = rays * depth[y, x, None]
    aa = xyz @ before[:3, :3].T + before[:3, 3]
    bb = xyz @ after[:3, :3].T + after[:3, 3]
    inside_a = ((aa >= bounds[0]) & (aa <= bounds[1])).all(1)
    inside_b = ((bb >= bounds[0]) & (bb <= bounds[1])).all(1)
    keep = inside_a | inside_b
    return dict(before=aa[keep].astype(np.float32), after=bb[keep].astype(np.float32),
                rgb=rgb[y[keep], x[keep]], pixels=np.column_stack([x[keep], y[keep]]).astype(np.int32))


def write_ply(path, clouds, variant, camera_colors=False):
    """Binary PLY retains every selected point, RGB and its camera_id (1/2)."""
    dtype = np.dtype([(v, '<f4') for v in ('x','y','z')] + [(v, 'u1') for v in ('red','green','blue','camera_id')])
    array = np.empty(sum(len(c[variant]) for c in clouds), dtype=dtype)
    offset = 0
    for i, cloud in enumerate(clouds):
        points = cloud[variant]
        sl = slice(offset, offset + len(points))
        for j, key in enumerate(('x','y','z')):
            array[key][sl] = points[:, j]
        colors = cloud['rgb'] if not camera_colors else np.tile(
            tuple(bytes.fromhex(VIEW_COLORS[i].lstrip('#'))), (len(points),1))
        for j, key in enumerate(('red','green','blue')):
            array[key][sl] = colors[:, j]
        array['camera_id'][sl] = i + 1
        offset += len(points)
    header = ('ply\nformat binary_little_endian 1.0\n'
              'comment Points are transformed observations; no ICP or surface reconstruction\n'
              f'element vertex {len(array)}\nproperty float x\nproperty float y\nproperty float z\n'
              'property uchar red\nproperty uchar green\nproperty uchar blue\nproperty uchar camera_id\nend_header\n')
    with Path(path).open('wb') as f:
        f.write(header.encode('ascii'))
        f.write(array.tobytes())


def static_plots(clouds, output, episode, frame, detail_bounds=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    from matplotlib.lines import Line2D
    # Both columns have the same physical ranges, projection and viewpoints.
    bounds = np.array([WORKSPACE_MIN, WORKSPACE_MAX]).copy()
    full = np.concatenate([c[key] for c in clouds for key in ('before','after')])
    bounds[0] = np.maximum(bounds[0], np.quantile(full, .002, axis=0) - .015)
    bounds[1] = np.minimum(bounds[1], np.quantile(full, .998, axis=0) + .015)
    for camera_colors in (True, False):
        fig = plt.figure(figsize=(14,10), layout='constrained')
        for row, (elev, azim, view) in enumerate(((26,-65,'Oblique'),(4,-85,'Low side view'))):
            for col, (variant, title) in enumerate((('before','Before refine'),('after','After refine'))):
                ax = fig.add_subplot(2,2,row*2+col+1, projection='3d')
                p = np.concatenate([c[variant] for c in clouds])
                colors = np.concatenate([np.tile(to_rgb(VIEW_COLORS[i]), (len(c[variant]),1))
                    if camera_colors else c['rgb'] / 255. for i,c in enumerate(clouds)])
                # One collection sorts BOTH cameras by depth; separate artists can hide
                # one camera wholesale, creating a misleading impression of alignment.
                ax.scatter(p[:,0],p[:,1],p[:,2],c=colors,s=.18,alpha=.85,depthshade=False,
                           linewidths=0,rasterized=True)
                ax.set(xlim=bounds[:,0],ylim=bounds[:,1],zlim=bounds[:,2],
                       xlabel='X (m)',ylabel='Y (m)',zlabel='Z (m)')
                ax.set_box_aspect(bounds[1]-bounds[0])
                ax.view_init(elev=elev,azim=azim)
                ax.set_proj_type('ortho')
                ax.tick_params(labelsize=8,pad=0)
                ax.set_title(f'{title} | {view}',fontsize=14,pad=0)
        fig.suptitle(f'Episode {episode}, frame {frame} | two external cameras in robot-base coordinates',fontsize=15)
        if camera_colors:
            fig.legend(handles=[Line2D([0],[0],marker='o',color='w',markerfacecolor=c,label=f'Camera {i+1}',markersize=9)
                                for i,c in enumerate(VIEW_COLORS)],loc='outside lower center',ncol=2)
        name = 'fusion_by_camera' if camera_colors else 'fusion_rgb'
        fig.savefig(output / f'{name}.png',dpi=160)
        fig.savefig(output / f'{name}.pdf')
        plt.close(fig)
    if detail_bounds is not None:
        # Select the union once, preserving point identity across the paired views.
        low, high = np.asarray(detail_bounds).reshape(2,3)
        require(np.all(high > low), 'Detail bounds must have max > min on all axes')
        selected = []
        for c in clouds:
            mask = np.zeros(len(c['before']),dtype=bool)
            for key in ('before','after'):
                mask |= ((c[key] >= low) & (c[key] <= high)).all(1)
            selected.append({k:v[mask] for k,v in c.items()})
        require(all(len(c['before']) for c in selected), 'Detail box has no points from one camera')
        colors = np.concatenate([np.tile(to_rgb(VIEW_COLORS[i]),(len(c['before']),1))
                                 for i,c in enumerate(selected)])
        # Fixed interleaving removes camera draw-order bias in the 2D projections.
        order = np.random.default_rng(0).permutation(len(colors))
        fig, axes = plt.subplots(2,2,figsize=(13,9),layout='constrained')
        for row, (dims, title) in enumerate((((0,1),'Top view'),((0,2),'Side view'))):
            for col, key in enumerate(('before','after')):
                ax = axes[row,col]
                p = np.concatenate([c[key] for c in selected])
                ax.scatter(p[order,dims[0]],p[order,dims[1]],c=colors[order],s=.35,alpha=.45,linewidths=0,rasterized=True)
                ax.set(xlim=(low[dims[0]],high[dims[0]]),ylim=(low[dims[1]],high[dims[1]]),
                       xlabel='XYZ'[dims[0]]+' (m)',ylabel='XYZ'[dims[1]]+' (m)',
                       title=f'{"Before" if col==0 else "After"} refine | {title}')
                ax.set_aspect('equal',adjustable='box')
                ax.grid(alpha=.15)
        fig.suptitle(f'Episode {episode}, frame {frame} | shared detail crop, identical source pixels',fontsize=15)
        fig.legend(handles=[Line2D([0],[0],marker='o',color='w',markerfacecolor=c,label=f'Camera {i+1}',markersize=9)
                            for i,c in enumerate(VIEW_COLORS)],loc='outside lower center',ncol=2)
        fig.savefig(output/'fusion_detail.png',dpi=180)
        fig.savefig(output/'fusion_detail.pdf')
        plt.close(fig)
    return bounds


def interactive_plot(clouds, output, episode, frame, after_name, bounds):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    after_label = {'recam_candidate':'PointWorld 方法优化',
                   'pointworld_release':'PointWorld 发布外参'}[after_name]
    fig = make_subplots(rows=1,cols=2,specs=[[{'type':'scene'},{'type':'scene'}]],
                        subplot_titles=('优化前：DROID 初值','优化后：'+after_label),horizontal_spacing=.03)
    rgb_colors, camera_colors = [], []
    for col, variant in enumerate(('before','after'),1):
        for i, cloud in enumerate(clouds):
            p = cloud[variant]
            rgb = [f'rgb({r},{g},{b})' for r,g,b in cloud['rgb']]
            rgb_colors.append(rgb)
            camera_colors.append(VIEW_COLORS[i])
            fig.add_trace(go.Scatter3d(x=p[:,0],y=p[:,1],z=p[:,2],mode='markers',name=f'相机 {i+1}',
                legendgroup=str(i),showlegend=col==1,hoverinfo='skip',
                marker=dict(size=1.4,color=VIEW_COLORS[i],opacity=1)),row=1,col=col)
    camera = dict(eye=dict(x=1.35,y=-1.8,z=.95),up=dict(x=0,y=0,z=1),projection=dict(type='orthographic'))
    scene = dict(camera=camera,aspectmode='data',dragmode='orbit',
                 xaxis=dict(title='X / m',range=bounds[:,0].tolist()),
                 yaxis=dict(title='Y / m',range=bounds[:,1].tolist()),
                 zaxis=dict(title='Z / m',range=bounds[:,2].tolist()))
    fig.update_layout(scene=scene,scene2=scene,height=720,margin=dict(l=0,r=0,t=90,b=10),
        paper_bgcolor='#f4f7fb',legend=dict(orientation='h',x=.5,xanchor='center',y=1.08),
        updatemenus=[dict(type='buttons',direction='right',x=0,y=1.12,
            buttons=[dict(label='按相机着色',method='restyle',args=[{'marker.color':camera_colors}]),
                     dict(label='真实 RGB',method='restyle',args=[{'marker.color':rgb_colors}])])])
    sync = """
    const plot=document.getElementById('{plot_id}');
    function fitViewport(){Plotly.relayout(plot,{height:Math.max(420,Math.min(720,window.innerHeight-220))});}
    fitViewport();
    window.addEventListener('resize',fitViewport);
    let syncing=false;
    plot.on('plotly_relayout',async function(event){
      if(syncing) return;
      let source=event['scene.camera']?'scene':event['scene2.camera']?'scene2':null;
      if(!source) return;
      syncing=true;
      try {await Plotly.relayout(plot,{[(source==='scene'?'scene2':'scene')+'.camera']:event[source+'.camera']});}
      finally {syncing=false;}
    });
    """
    plot = fig.to_html(full_html=False,include_plotlyjs=True,post_script=sync,
                       config=dict(displaylogo=False,scrollZoom=True,responsive=True),div_id='fusion-view')
    title = f'Episode {episode} · 第 {frame} 帧 · 双视角融合'
    detail = ('<details open><summary>桌面与物体局部：相同裁剪区域</summary>'
              '<img class="comparison" src="fusion_detail.png" alt="局部点云：左优化前、右优化后"></details>'
              if (output/'fusion_detail.png').exists() else '')
    text = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>{html.escape(title)}</title>
    <style>body{{margin:0;background:#f4f7fb;color:#17283d;font:16px system-ui;overflow-x:hidden}}header,footer{{padding:18px 30px}}
    h1{{font-size:25px;margin:0 0 10px}}p{{line-height:1.6;margin:6px 0}}a{{color:#075ea8}}
    summary{{cursor:pointer;font-weight:600;margin:18px 0}}.comparison{{width:100%;max-width:1400px}}
    .sources{{display:flex;gap:16px}}.sources img{{width:calc(50% - 8px);object-fit:contain}}</style>
    <header><h1>{html.escape(title)}</h1><p>拖动旋转、滚轮缩放，左右视角同步；点击“真实 RGB”切换颜色。</p>
    <p>红色为相机 1，青色为相机 2。重点观察杯口、物体轮廓、桌面和机械臂是否出现两层或错位。</p>
    <p>前后使用完全相同的深度、内参和源像素，仅更换外参。保留机械臂；没有 ICP、平滑、补面或合并平均。</p></header>
    {plot}<footer><p>坐标单位为米。每台相机的点都保留在各自可见表面上；单视角遮挡区域无需完全重合。</p>
    <p><a href="before_rgb.ply">优化前 PLY</a> · <a href="after_rgb.ply">优化后 PLY</a> ·
    <a href="fusion_by_camera.png">按相机着色的固定视图</a> · <a href="fusion_rgb.png">RGB 固定视图</a> ·
    <a href="provenance.json">数据与校验记录</a></p>{detail}
    <details><summary>原始 RGB：左相机 1，右相机 2</summary><div class="sources">
    <img src="source_camera1.png" alt="原始相机 1 RGB"><img src="source_camera2.png" alt="原始相机 2 RGB"></div></details>
    </footer></html>"""
    (output/'index.html').write_text(text,encoding='utf-8')


def run_fusion(args):
    root, output = args.root.resolve(), args.output_dir.resolve()
    require(not output.is_relative_to(root), 'Fusion output must be outside the dataset')
    if (root/'real_world/droid').is_dir():
        root = root/'real_world/droid'
    report = read_json(args.metrics)
    require(output != args.metrics.resolve().parent, 'Use a separate output directory; preserve the audit report')
    episode = int(report['episode_index'])
    representatives = report['image_frames'] or report['evaluation_frames']
    require(representatives, 'Audit has no evaluation frames')
    frame = args.frame if args.frame is not None else representatives[len(representatives)//2]
    require(frame in report['evaluation_frames'], 'Choose a frame in the existing audit evaluation_frames')
    after_name = args.after or ('recam_candidate' if 'recam_candidate' in report['poses'] else 'pointworld_release')
    require(after_name in report['poses'], f'Missing audited variant: {after_name}')
    detail_bounds = getattr(args,'detail_bounds',None)
    if (output/'provenance.json').exists():
        previous = read_json(output/'provenance.json')
        require(all(previous.get(k)==v for k,v in dict(episode_index=episode,frame_index=frame,
                after=after_name,detail_bounds=detail_bounds).items()),
                'Different episode/frame/variant/crop already exported here; choose a new output directory')
    before = check_transform(report['poses']['droid_initial'],'before')
    after = check_transform(report['poses'][after_name],'after')
    require(before.shape == after.shape == (2,4,4), 'Two external camera poses required')
    kk = np.asarray(report['intrinsics_half_resolution'])
    info = read_json(root/'meta/info.json')
    output.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(output/'.matplotlib'))
    hashes = {str(args.metrics.resolve()):sha256(args.metrics)}
    clouds = []
    for c in (1,2):
        png = media_path(root,info,episode,f'observation.images.depth_{c:02d}',frame)
        video = media_path(root,info,episode,f'observation.images.rgb_{c:02d}')
        for p in (png,video):
            hashes[str(p)] = sha256(p)
            expected = report['input_sha256'].get(str(p))
            require(expected is not None and hashes[str(p)] == expected, f'Input differs from audited data: {p}')
        with Image.open(png) as im:
            depth = np.asarray(im,dtype=np.float32)[::2,::2]/1000.
        rgb = read_rgb(video,[frame])[frame]
        Image.fromarray(rgb).save(output/f'source_camera{c}.png')
        clouds.append(paired_cloud(depth,rgb,kk[c-1],before[c-1],after[c-1]))
    require(all(len(c['before']) for c in clouds), 'No workspace points from an external camera')
    for variant in ('before','after'):
        write_ply(output/f'{variant}_rgb.ply',clouds,variant)
        write_ply(output/f'{variant}_by_camera.ply',clouds,variant,True)
    np.savez_compressed(output/'paired_clouds.npz',**{f'camera{i+1}_{key}':value for i,c in enumerate(clouds) for key,value in c.items()})
    bounds = static_plots(clouds,output,episode,frame,detail_bounds)
    interactive_plot(clouds,output,episode,frame,after_name,bounds)
    require(all(sha256(Path(p))==h for p,h in hashes.items()), 'Input changed during fusion export')
    write_json(output/'provenance.json',dict(episode_index=episode,frame_index=frame,after=after_name,
        points_per_camera=[len(c['before']) for c in clouds],same_source_pixels=True,include_robot=True,
        depth_scale='half resolution, millimeter PNG converted to meters',
        crop='union of before/after workspace membership, identical pixels in both panels',
        workspace_min=WORKSPACE_MIN.tolist(),workspace_max=WORKSPACE_MAX.tolist(),
        display_bounds=bounds.tolist(),source_sha256=hashes,inputs_unchanged=True,
        detail_bounds=detail_bounds,
        artifact_sha256={p.name:sha256(p) for p in output.iterdir() if p.suffix in ('.png','.ply','.html','.npz','.pdf')}))
    print(f'FUSION {output / "index.html"}',flush=True)
