from pathlib import Path
import tempfile
import unittest

import numpy as np

from recam_refine.fusion import paired_cloud, write_ply


class FusionTests(unittest.TestCase):
    def test_pose_change_never_changes_selected_pixel_identity(self):
        # A pixel leaves the ROI, a second enters it, a third stays outside.
        # Both paired clouds must keep the first two, including the now-outside point.
        depth = np.array([[1.,1.,1.,0.,np.nan,5.]])
        rgb = np.arange(18,dtype=np.uint8).reshape(1,6,3)
        after = np.eye(4)
        after[0,3] = -1
        cloud = paired_cloud(depth,rgb,np.eye(3),np.eye(4),after,
                             bounds=(np.array([-.1,-.1,.5]),np.array([.1,.1,1.5])))
        np.testing.assert_array_equal(cloud['pixels'],[[0,0],[1,0]])
        np.testing.assert_array_equal(cloud['rgb'],[[0,1,2],[3,4,5]])
        np.testing.assert_allclose(cloud['before'],[[0,0,1],[1,0,1]])
        np.testing.assert_allclose(cloud['after'],[[-1,0,1],[0,0,1]])

    def test_ply_retains_original_color_geometry_and_camera_identity(self):
        clouds = [dict(before=np.array([[1,2,3]],dtype=np.float32),rgb=np.array([[4,5,6]],dtype=np.uint8)),
                  dict(before=np.array([[7,8,9]],dtype=np.float32),rgb=np.array([[10,11,12]],dtype=np.uint8))]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'cloud.ply'
            write_ply(path,clouds,'before')
            header, data = path.read_bytes().split(b'end_header\n',1)
            self.assertIn(b'element vertex 2',header)
            self.assertEqual(len(data),32)
            import struct
            self.assertEqual(struct.unpack('<fffBBBB',data[:16]),(1,2,3,4,5,6,1))
            self.assertEqual(struct.unpack('<fffBBBB',data[16:]),(7,8,9,10,11,12,2))


if __name__ == '__main__':
    unittest.main()
