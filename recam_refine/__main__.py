"""python -m recam_refine --help"""
import argparse
import json
import os
from pathlib import Path
import platform
import sys


def doctor(gpu=False,torch_cpu=False):
    import av
    import numpy as np
    import pyarrow
    from PIL import Image
    from .common import require
    require(sys.version_info[:2] in ((3, 10), (3, 11)), "Use Python 3.10 or 3.11")
    av.codec.Codec("libx264", "w")
    result = dict(python=platform.python_version(), platform=platform.platform(),
                  av=av.__version__, pyarrow=pyarrow.__version__, numpy=np.__version__, pillow=Image.__version__)
    if torch_cpu:
        import torch
        d = torch.ones((1,1,8,8),device='cpu')
        g = torch.zeros((1,1,4,2),device='cpu',requires_grad=True)
        torch.nn.functional.grid_sample(d,g,align_corners=True).sum().backward()
        require(g.grad is not None and bool(torch.isfinite(g.grad).all()),'CPU grid_sample check failed')
        result.update(torch=torch.__version__,torch_cpu_check=True)
    if gpu:
        import torch
        from .batched import CameraBatch
        require(torch.cuda.is_available(), "No CUDA GPU accessible; check NVIDIA driver and CUDA_VISIBLE_DEVICES")
        devices = []
        graphs = []
        for i in range(torch.cuda.device_count()):
            # Exercise grid_sample forward/backward, the actual calibration
            # operation, not just CUDA availability. No CUDA toolkit is needed.
            d = torch.ones((1,1,8,8), device=f"cuda:{i}")
            g = torch.zeros((1,1,4,2), device=f"cuda:{i}", requires_grad=True)
            torch.nn.functional.grid_sample(d, g, align_corners=True).sum().backward()
            torch.cuda.synchronize(i)
            xy = np.stack(np.meshgrid(np.linspace(-.2,.2,10),np.linspace(-.2,.2,10)),-1).reshape(-1,2)
            points = np.column_stack([xy,np.ones(len(xy))]).astype(np.float32)
            fixture = dict(initial=np.eye(4),k=np.array([[50,0,32],[0,50,24],[0,0,1]]),
                           depths=[np.ones((48,64),np.float32)]*4,points=[points]*4)
            batch = CameraBatch([fixture],f'cuda:{i}',min_points=10)
            batch.advance(2)
            require(batch.finish()[0][1]['accepted'],f'Batched CUDA optimizer check failed on GPU {i}')
            graphs.append(batch.graph is not None)
            del batch
            devices.append(torch.cuda.get_device_name(i))
        result.update(torch=torch.__version__, cuda=torch.version.cuda, devices=devices,batched_cuda_graphs=graphs)
    print(json.dumps(result, indent=2), flush=True)


def calibration_options(parser):
    parser.add_argument('--refine-backend',choices=('auto','batched','reference'),default='auto',
                        help='New CUDA runs use batched FP32; existing work directories retain their saved backend')
    parser.add_argument('--gpu-batch-size',type=int,default=0,
                        help='Maximum cameras per GPU batch (two per episode); 0 sizes from free VRAM with OOM backoff')
    parser.add_argument('--no-cuda-graphs',action='store_true',
                        help='Use eager batched GPU execution; same inputs, budget and quality gates')


class ExplicitIterations(argparse.Action):
    def __call__(self,parser,namespace,values,option_string=None):
        setattr(namespace,self.dest,values)
        namespace.iterations_explicit = True


def main():
    parser = argparse.ArgumentParser(description="Recover, align, refine and fully check ReCam LeRobot datasets.")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("doctor", help="Test runtime and codecs without touching datasets")
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--torch-cpu", action="store_true")
    p = sub.add_parser("run", help="Run/resume the complete offline pipeline")
    p.add_argument("root", type=Path, help="recam_lerobot directory")
    p.add_argument("--work-dir", type=Path, required=True, help="Separate directory for state, backups, logs and reports")
    p.add_argument("--depth-output", type=Path, help="Completed droid-metric-depth output root; includes images/ and annotations/")
    p.add_argument("--depth-chunks", default="2-13")
    p.add_argument("--depth-metadata", type=Path, nargs="*", default=[], help="Additional FoundationStereo sidecar/output roots")
    p.add_argument("--episode-manifest", type=Path, help="Original episode_manifest.jsonl or directory of chunk JSONLs; auto-download if omitted")
    p.add_argument("--pointworld-cameras", type=Path, help="Directory containing UUID_cameras.json; auto-download if omitted")
    p.add_argument("--workers", type=int, default=4, help="CPU/I/O processes; default 4 avoids overwhelming shared storage")
    p.add_argument("--devices", default="0,1,2,3,4,5,6,7", help="GPU IDs, e.g. 0,1 or cpu for tests")
    p.add_argument("--iterations", type=int, default=2000)
    calibration_options(p)
    p.add_argument("--defer-cleanup", action="store_true", help="Stop after full media checks so the entry script can audit geometry before cleanup")
    p = sub.add_parser('transfer-depth',help='Step 1 only: move completed metric depth, verify it, then stop')
    p.add_argument('root',type=Path)
    p.add_argument('--work-dir',type=Path,required=True)
    p.add_argument('--depth-output',type=Path,required=True)
    p.add_argument('--depth-chunks',default='2-13')
    p.add_argument('--episode-manifest',type=Path)
    p = sub.add_parser('run-step',help='Run exactly one manual stage, then stop')
    p.add_argument('step',choices=('unpack','align','overlap','refine','apply','check','cleanup','repack','status',
                                  'shard-plan','shard-refine','shard-merge','shard-status'))
    p.add_argument('root',type=Path)
    p.add_argument('--work-dir',type=Path,required=True)
    p.add_argument('--episode-manifest',type=Path)
    p.add_argument('--depth-metadata',type=Path,nargs='*',default=[])
    p.add_argument('--pointworld-cameras',type=Path)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--devices',default='0,1,2,3,4,5,6,7')
    p.add_argument('--iterations',type=int,default=2000,action=ExplicitIterations)
    calibration_options(p)
    p.add_argument('--audit-frames',type=int,default=8)
    p.add_argument('--num-shards',type=int,help='shard-plan: number of fixed episode partitions')
    p.add_argument('--shard-id',type=int,help='shard-refine: zero-based partition to compute')
    p.add_argument('--worker-work-dir',type=Path,help='shard-refine: separate worker logs/checkpoints/cache; Bash --runtime-work-dir selects a shared environment')
    p.add_argument('--episodes-per-shard',type=int,default=250,help='repack: maximum episodes per external-depth TAR')
    p = sub.add_parser("check", help="Read-only full media decoding and metadata validation")
    p.add_argument("root", type=Path)
    p.add_argument("--report-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    p = sub.add_parser("overlap", help="Read-only exact source UUID + camera serial overlap report")
    p.add_argument("--episode-manifest", type=Path, required=True)
    p.add_argument("--pointworld-cameras", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p = sub.add_parser("audit-cameras", help="Read-only PointWorld metric comparison and RGB/point-cloud report")
    p.add_argument("root", type=Path, help="DROID subset or recam_lerobot root")
    p.add_argument("--work-dir", type=Path, required=True, help="Pipeline work directory (contains initial plan and poses)")
    p.add_argument("--report-dir", type=Path, required=True, help="HTML/PNG/JSON output outside the dataset")
    p.add_argument("--episodes", nargs="+", required=True, help="Episode indices, or all")
    p.add_argument("--episode-manifest", type=Path)
    p.add_argument("--pointworld-cameras", type=Path)
    p.add_argument("--candidate-dir", type=Path, help="Defaults to work-dir/cameras")
    p.add_argument("--depth-metadata", type=Path, nargs="*", default=[])
    p.add_argument("--frames", type=int, default=24, help="Evaluation frames outside all fitting/selection frames")
    p.add_argument("--image-frames", type=int, default=3, help="Representative image frames per episode; 0 for full-dataset metric scans")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--fail-on-review", action="store_true", help="Return exit code 2 when geometry is unavailable, regresses, or exceeds depth threshold")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--fit", action="store_true", help="Fit both external cameras in report directory without modifying the dataset")
    p.add_argument("--iterations", type=int, default=2000)
    p = sub.add_parser("visualize-fusion", help="Read-only before/after two-view point clouds, PLY and synchronized offline 3D viewer")
    p.add_argument("root", type=Path)
    p.add_argument("--metrics", type=Path, required=True, help="An episode's camera audit metrics.json")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--frame", type=int, help="Audited frame; defaults to the middle representative image frame")
    p.add_argument("--after", choices=("recam_candidate", "pointworld_release"))
    p.add_argument("--detail-bounds", type=float, nargs=6, metavar=("XMIN","YMIN","ZMIN","XMAX","YMAX","ZMAX"),
                   help="Optional shared detail crop in robot-base meters")
    p = sub.add_parser("prepare-viewer", help="Package audited point clouds and the matching URDF pose for Viser")
    p.add_argument("root", type=Path)
    p.add_argument("--fusion-dir", type=Path, required=True)
    p.add_argument("--metrics", type=Path, required=True)
    p.add_argument("--robot-urdf", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    if getattr(args, "workers", 1) < 1 or getattr(args, "iterations", 1) < 1:
        parser.error("workers and iterations must be positive")
    if getattr(args,'gpu_batch_size',0) < 0:
        parser.error('gpu-batch-size must be zero (auto) or positive')
    try:
        if args.command == "doctor":
            doctor(args.gpu,args.torch_cpu)
        elif args.command == "run":
            from .pipeline import run
            run(args)
        elif args.command == 'transfer-depth':
            from .steps import transfer_only
            transfer_only(args)
        elif args.command == 'run-step':
            if args.step == 'repack':
                from .repack import run_repack
                run_repack(args)
            else:
                from .steps import run_step
                run_step(args)
        elif args.command == "check":
            from .pipeline import discover
            from .validate import check_subset
            from .common import require, write_json
            require(not args.report_dir.resolve().is_relative_to(args.root.resolve()), "Report directory must be outside the dataset")
            (args.report_dir / "SUCCESS.json").unlink(missing_ok=True)
            results = [check_subset(s, args.report_dir, args.workers, droid=s.name == "droid") for s in discover(args.root)]
            write_json(args.report_dir / "SUCCESS.json", results)
        elif args.command == "overlap":
            from collections import Counter
            from .common import read_jsonl, write_json
            from .pointworld import release_pose
            rows = read_jsonl(args.episode_manifest)
            counts, details = Counter(), []
            for row in rows:
                if "camera_serials" not in row:
                    row["camera_serials"] = {role:str(v["serial"]) for role,v in row["cameras"].items()}
                _, status = release_pose(args.pointworld_cameras, row)
                counts[status] += 1
                details.append(dict(episode_index=row["episode_index"], source_episode_id=row["source_episode_id"], status=status))
            write_json(args.report, dict(counts=dict(counts), episodes=details))
            print(json.dumps(dict(counts)))
        elif args.command == "audit-cameras":
            from .audit import run_audit
            return run_audit(args)
        elif args.command == "visualize-fusion":
            from .fusion import run_fusion
            run_fusion(args)
        elif args.command == "prepare-viewer":
            from .viewer_bundle import prepare_viewer
            prepare_viewer(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        if args.command in ('run','transfer-depth','run-step') and not args.work_dir.resolve().is_relative_to(args.root.resolve()):
            import traceback
            from .common import write_json
            name = ('FAILED.json' if args.command=='run' else 'STEP1_FAILED.json' if args.command=='transfer-depth'
                    else args.step.upper()+'_FAILED.json')
            write_json(args.work_dir / name,
                       dict(error=str(exc), traceback=traceback.format_exc()))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
