"""Incremental validated sidecar cache, stored outside the training dataset."""
import hashlib
import json
from pathlib import Path
import sqlite3


class InputCache:
    def __init__(self, directory, manifest):
        self.db = None
        self.hits = self.misses = 0
        self.stamps = {}
        self.identity = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        if directory is not None:
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)
            self.db = sqlite3.connect(directory/'validated_depth_json_v1.sqlite3', timeout=60)
            self.db.execute('CREATE TABLE IF NOT EXISTS records (path TEXT, identity TEXT, stamp TEXT, value TEXT, PRIMARY KEY(path, identity))')
            self.db.commit()

    @staticmethod
    def stamp(path):
        s = path.stat()
        return json.dumps([s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns])

    def get(self, path):
        if self.db is None:
            return None
        stamp = self.stamp(path)
        self.stamps[str(path)] = stamp
        row = self.db.execute('SELECT stamp,value FROM records WHERE path=? AND identity=?',
                              (str(path), self.identity)).fetchone()
        if row and row[0] == stamp:
            self.hits += 1
            return json.loads(row[1])
        self.misses += 1
        return None

    def put(self, path, key, value):
        if self.db is None:
            return
        stamp = self.stamp(path)
        if stamp != self.stamps.get(str(path)):
            raise RuntimeError(f'Depth JSON changed while being read: {path}')
        self.db.execute('INSERT OR REPLACE INTO records VALUES (?,?,?,?)',
                        (str(path), self.identity, stamp, json.dumps([key, value])))
        if self.misses % 64 == 0:
            self.db.commit()

    def close(self):
        if self.db is not None:
            self.db.commit()
            self.db.close()
