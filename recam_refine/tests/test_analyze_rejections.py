import json
from pathlib import Path
import tempfile
import unittest

from recam_refine.analyze_rejections import analyze, classify


class RejectionTests(unittest.TestCase):
    def metric(self, **changes):
        return dict(dict(accepted=False,initial_train_loss=.2,final_train_loss=.1,
            initial_holdout_loss=.15,final_holdout_loss=.11,
            translation_change_m=.1,rotation_change_deg=5),**changes)

    def test_missing_metrics_and_overlapping_conditions(self):
        _, reasons, improved, complete = classify(self.metric())
        self.assertEqual(reasons,['验证误差达到或超过10cm'])
        self.assertTrue(improved and complete)
        _, reasons, improved, _ = classify(self.metric(final_holdout_loss=.2,translation_change_m=.4))
        self.assertEqual(len(reasons),3)
        self.assertFalse(improved)
        _, reasons, improved, complete = classify({'failed_metrics':{'final_holdout_loss':None}})
        self.assertIn('指标缺失或无有效数值',reasons)
        self.assertFalse(improved or complete)

    def test_both_shards_skip_accepted_and_excluded(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            for shard in (0,1):
                directory=root/f'shard_{shard}'/'cameras'
                directory.mkdir(parents=True)
                (directory/'episode_000000.json').write_text(json.dumps(dict(episode_index=shard,
                    metrics=[self.metric(),self.metric(accepted=True)])))
                (directory/'episode_000001.json').write_text(json.dumps(dict(excluded_bad_depth=True)))
            result=analyze(root,root/'report.json')
            self.assertEqual(result['summary']['保留原值的相机总数'],2)
            self.assertEqual(len(result['cameras']),2)
            self.assertEqual(json.loads((root/'report.json').read_text(encoding='utf-8')),result)
