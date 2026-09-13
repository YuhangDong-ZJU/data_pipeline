import unittest
from recam_refine.calibration_report import CalibrationReport


class ReportTests(unittest.TestCase):
    def test_camera_and_episode_counts_are_distinct(self):
        report=CalibrationReport()
        for decisions in ((True,True),(True,False),(False,False)):
            report.add(dict(source='pointworld_method_droid_initialization',
                            metrics=[dict(accepted=v) for v in decisions]))
        report.add(dict(source='pointworld_release',camera_to_base=[[],[]]))
        report.add(dict(excluded_bad_depth=True))
        c=report.counts
        self.assertEqual(c['result_episodes'],5)
        self.assertEqual(c['optimized_accepted_cameras'],3)
        self.assertEqual(c['retained_initial_cameras'],3)
        self.assertEqual(c['pointworld_release_cameras'],2)
        self.assertEqual(c['excluded_bad_depth_episodes'],1)
        for key in ('both_cameras_accepted_episodes','mixed_acceptance_episodes','both_cameras_retained_episodes'):
            self.assertEqual(c[key],1)
        self.assertEqual(c['unknown_cameras'],0)

    def test_missing_decisions_are_not_counted_as_acceptance(self):
        report=CalibrationReport()
        report.add(dict(source='pointworld_method_droid_initialization',metrics=[{}]))
        self.assertEqual(report.counts['unknown_cameras'],2)
        self.assertEqual(report.counts['optimized_accepted_cameras'],0)
