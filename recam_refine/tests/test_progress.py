import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

from recam_refine.progress import run


class ProgressTests(unittest.TestCase):
    def test_standalone_scanner_logs_and_preserves_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'input'
            record = root/'annotations/foundation_stereo_depth/chunk-006/observation.images.depth_01/episode_006795.json'
            record.parent.mkdir(parents=True)
            record.write_text(json.dumps({'source': dict(episode_index=6795,
                frame_count=3, decoded_frame_count=3, tail_missing_count=0,
                missing_frame_indices=[], timestamps_ms=[1,2,2],
                initial_decoded_frame_count=2, tail_retry_recovered_frames=1,
                tail_retry_attempted=True)}), encoding='utf-8')
            original = record.read_bytes()
            report = Path(tmp)/'report.json'
            script = Path(__file__).resolve().parents[1]/'scan_depth_timestamps.py'
            result = subprocess.run([sys.executable, str(script), '--depth-output', str(root),
                                     '--report', str(report)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(report.read_text())['affected_episode_ids'], ['006795'])
            log = report.with_suffix('.log').read_text(encoding='utf-8')
            self.assertIn('affected episodes=1', log)
            self.assertIn('SUCCESS exit_code=0', log)
            self.assertEqual(original, record.read_bytes())

    def test_quiet_child_progress_capture_and_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp)/'stage.log'
            code = run([sys.executable, '-c',
                        "from recam_refine.progress import phase; import time; "
                        "phase('verification', 2, 5); time.sleep(.3); raise SystemExit(7)"],
                       'test', log, interval=.05, capture=True)
            self.assertEqual(code, 7)
            output = log.read_text(encoding='utf-8')
            self.assertIn('verification 2/5', output)
            self.assertIn('completed=2/5', output)
            self.assertIn('RUNNING pid=', output)
            self.assertIn('FAILED exit_code=7', output)
            self.assertFalse(list(Path(tmp).glob('.progress-*')))

    @unittest.skipUnless(os.name == 'posix', 'POSIX process-group signal forwarding')
    def test_interrupt_reaches_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            ready = Path(tmp)/'ready'
            stopped = Path(tmp)/'stopped'
            script = ("import signal,time; from pathlib import Path; "
                      f"signal.signal(signal.SIGINT, lambda *a: (Path({str(stopped)!r}).touch(), exit(130))); "
                      f"Path({str(ready)!r}).touch(); time.sleep(30)")
            process = subprocess.Popen([sys.executable, '-m', 'recam_refine.progress',
                                        '--label', 'signal', '--log', str(Path(tmp)/'stage.log'),
                                        '--', sys.executable, '-c', script],
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            try:
                deadline = time.monotonic()+10
                while not ready.exists() and time.monotonic()<deadline:
                    time.sleep(.02)
                self.assertTrue(ready.exists())
                process.send_signal(signal.SIGINT)
                output, _ = process.communicate(timeout=5)
                self.assertEqual(process.returncode, 130, output)
                self.assertTrue(stopped.exists())
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate()


if __name__ == '__main__':
    unittest.main()
