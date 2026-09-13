"""Summarize saved rejection metrics. Standard library only; no GPU/media reads."""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

DEFAULT_WORKERS = '/mnt/bn/yuyingchen/moranli/Code/Research/ModelArch/ReCam/recam_refine_workers'
DEFAULT_OUTPUT = '/mnt/bn/pistis/moranli/Data/recam_lerobot/recam_refine_work/rejection_analysis.json'
KEYS = ('initial_train_loss', 'final_train_loss', 'initial_holdout_loss',
        'final_holdout_loss', 'translation_change_m', 'rotation_change_deg')


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def classify(metric):
    m = metric.get('failed_metrics') or metric
    values = {key: m.get(key) for key in KEYS}
    reasons = []
    checks = (
        ('验证误差达到或超过10cm', ('final_holdout_loss',), lambda t: t >= .1),
        ('验证误差增加超过容差', ('initial_holdout_loss', 'final_holdout_loss'), lambda a,b: b > a+.0001),
        ('训练误差增加', ('initial_train_loss', 'final_train_loss'), lambda a,b: b > a),
        ('平移变化超过30cm', ('translation_change_m',), lambda t: t > .30),
        ('旋转变化超过25度', ('rotation_change_deg',), lambda t: t > 25),
    )
    for label, fields, test in checks:
        if all(number(values[k]) for k in fields) and test(*(values[k] for k in fields)):
            reasons.append(label)
    complete = all(number(v) for v in values.values())
    if not complete:
        reasons.append('指标缺失或无有效数值')
    elif not reasons:
        reasons.append('数值门槛均满足，但仍被拒绝；需确认可观测性')
    improved = (all(number(values[k]) for k in KEYS[:4])
                and values['final_train_loss'] <= values['initial_train_loss']
                and values['final_holdout_loss'] < values['initial_holdout_loss'])
    return values, reasons, improved, complete


def analyze(workers, output):
    counts, details = Counter(), []
    for shard in (0, 1):
        paths = sorted((workers/f'shard_{shard}'/'cameras').glob('episode_*.json'))
        if not paths:
            raise ValueError(f'未找到分片 {shard} 的结果：{workers / f"shard_{shard}" / "cameras"}')
        for n,path in enumerate(paths,1):
            try:
                result = json.loads(path.read_text(encoding='utf-8-sig'))
                if not result.get('excluded_bad_depth'):
                    for camera,metric in enumerate(result.get('metrics',[]),1):
                        if metric.get('accepted') is not False:
                            continue
                        counts['保留原值的相机总数'] += 1
                        values,reasons,improved,complete = classify(metric)
                        counts.update(reasons)
                        if improved:
                            counts['训练不变差且验证误差确实下降'] += 1
                            if complete and reasons == ['验证误差达到或超过10cm']:
                                counts['有改善且已知数值条件仅10cm不满足；可观测性待确认'] += 1
                        details.append(dict(shard=shard,episode_index=result['episode_index'],
                            source_episode_id=result.get('source_episode_id'),camera=camera,
                            recorded_reason=metric.get('reason'),detected_reasons=reasons,
                            improved=improved,**values))
            except Exception as exc:
                raise ValueError(f'读取结果失败：{path}: {exc}') from exc
            if n % 500 == 0 or n == len(paths):
                print(f'分片 {shard}：progress={n}/{len(paths)}',flush=True)
    report = dict(summary=dict(counts),
        note='统计已保存结果，可能不包含未完成的 episode。拒绝原因可重叠，不能相加；误差和平移单位为米，旋转为度。旧结果未保存完整可观测性标记；未重新判定或修改外参。',
        cameras=details)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('\n统计结果（原因可重叠）：',flush=True)
    for label,count in counts.items():
        print(f'{label}：{count}',flush=True)
    print(f'详细报告：{output}',flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers-root',type=Path,default=Path(DEFAULT_WORKERS))
    p.add_argument('--output',type=Path,default=Path(DEFAULT_OUTPUT))
    args = p.parse_args()
    try:
        analyze(args.workers_root,args.output)
    except (OSError,ValueError) as exc:
        p.exit(1,f'ERROR: {exc}\n')


if __name__ == '__main__':
    main()
