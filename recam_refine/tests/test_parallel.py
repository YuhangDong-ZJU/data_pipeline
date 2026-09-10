import threading
import time
import unittest
from unittest.mock import patch

from recam_refine.parallel import io_map


class ParallelTests(unittest.TestCase):
    def test_real_overlap_bounded_workers_and_ordered_results(self):
        barrier = threading.Barrier(2)
        lock = threading.Lock()
        running = peak = 0
        coordinator = threading.get_ident()
        def function(i):
            nonlocal running, peak
            with lock:
                running += 1
                peak = max(peak, running)
            if i < 2:
                barrier.wait(timeout=5)
            time.sleep(.01)
            with lock:
                running -= 1
            return i*2
        def report(*args):
            self.assertEqual(threading.get_ident(), coordinator)
        with patch('recam_refine.parallel.phase', side_effect=report):
            self.assertEqual(io_map(function, range(10), 2, 'test'), list(range(0,20,2)))
        self.assertEqual(peak, 2)

    def test_failed_work_is_not_reported_complete(self):
        def fail(i):
            raise ValueError('failed')
        with patch('recam_refine.parallel.phase') as report:
            with self.assertRaisesRegex(ValueError, 'failed'):
                io_map(fail, [1,2,3], 2, 'test')
            self.assertFalse(any(call.args[1:3] == (3,3) for call in report.call_args_list))
