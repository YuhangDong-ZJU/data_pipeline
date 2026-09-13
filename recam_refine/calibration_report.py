"""Summarize saved camera decisions without reading media or rerunning fitting."""
import argparse
from pathlib import Path

from .common import read_json, write_json


class CalibrationReport:
    def __init__(self):
        self.counts = dict(result_episodes=0, optimized_accepted_cameras=0,
                          retained_initial_cameras=0, pointworld_release_cameras=0,
                          excluded_bad_depth_episodes=0, both_cameras_accepted_episodes=0,
                          mixed_acceptance_episodes=0, both_cameras_retained_episodes=0,
                          unknown_cameras=0)

    def add(self, result):
        c = self.counts
        c['result_episodes'] += 1
        if result.get('excluded_bad_depth'):
            c['excluded_bad_depth_episodes'] += 1
            return
        if result.get('source') != 'pointworld_method_droid_initialization':
            # Release candidates have two poses and no local fit metrics.
            if str(result.get('source','')).startswith('pointworld') and len(result.get('camera_to_base',[]))==2:
                c['pointworld_release_cameras'] += 2
            else:
                c['unknown_cameras'] += 2
            return
        decisions = [m.get('accepted') for m in result.get('metrics',[])[:2]]
        accepted = sum(v is True for v in decisions)
        retained = sum(v is False for v in decisions)
        c['optimized_accepted_cameras'] += accepted
        c['retained_initial_cameras'] += retained
        c['unknown_cameras'] += 2-accepted-retained
        if accepted==2:
            c['both_cameras_accepted_episodes'] += 1
        elif retained==2:
            c['both_cameras_retained_episodes'] += 1
        elif accepted==retained==1:
            c['mixed_acceptance_episodes'] += 1

    def save(self, directory, scope):
        path = Path(directory)/'calibration_acceptance_report.json'
        write_json(path,dict(scope=scope,**self.counts,
            note='Counts cover saved episode results only. Camera counts exclude wrist and bad-depth episodes. PointWorld releases are separate from local optimization.'))
        c = self.counts
        print(f"外参验收汇总 [{scope}]：优化通过={c['optimized_accepted_cameras']} 个相机；"
              f"未通过、保留原值={c['retained_initial_cameras']} 个相机；"
              f"PointWorld 直接复用={c['pointworld_release_cameras']} 个相机；"
              f"坏图排除={c['excluded_bad_depth_episodes']} 个 episode；"
              f"未分类={c['unknown_cameras']} 个相机。报告：{path}",flush=True)
        return self.counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('work_dir',type=Path,help='Coordinator or individual worker work directory')
    args = parser.parse_args()
    report = CalibrationReport()
    paths = sorted((args.work_dir/'cameras').glob('episode_*.json'))
    if not paths:
        parser.error(f'No saved results in {args.work_dir / "cameras"}')
    for path in paths:
        report.add(read_json(path))
    report.save(args.work_dir,'saved results; may be incomplete')


if __name__=='__main__':
    main()
