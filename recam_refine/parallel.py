"""Bounded I/O concurrency; only the coordinator writes progress state."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import threading
import time

from .progress import phase


class Counter:
    def __init__(self):
        self.value = 0
        self.lock = threading.Lock()

    def tick(self, amount=1):
        with self.lock:
            self.value += amount


def io_map(function, items, workers, name, detail=None):
    if workers < 1:
        raise ValueError('I/O workers must be positive')
    items = list(items)
    results = [None] * len(items)
    completed = submitted = 0
    describe = lambda: detail() if detail else f'并发={workers}'
    phase(name, 0, len(items), describe())
    updated = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {}
        try:
            while completed < len(items):
                while submitted < len(items) and len(pending) < workers * 2:
                    pending[pool.submit(function, items[submitted])] = submitted
                    submitted += 1
                done, _ = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
                for future in done:
                    results[pending.pop(future)] = future.result()
                    completed += 1
                now = time.monotonic()
                if now - updated >= 1 or completed == len(items):
                    phase(name, completed, len(items), describe())
                    updated = now
        except BaseException:
            for future in pending:
                future.cancel()
            raise
    return results
