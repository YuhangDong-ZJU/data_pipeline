"""Numerically stable, mergeable population statistics."""
from __future__ import annotations

import numpy as np

from .common import require, values


class Moments:
    def __init__(self):
        self.n = 0
        self.frames = 0

    def add(self, a, frames=None):
        a = np.asarray(a, dtype=np.float64)
        require(a.size and np.isfinite(a).all(), "Empty/non-finite data in statistics")
        n = len(a)
        mean = a.mean(axis=0)
        var = a.var(axis=0)
        self.merge(n, mean, var * n, a.min(axis=0), a.max(axis=0), n if frames is None else frames)

    def merge(self, n, mean, m2, lo, hi, frames):
        if self.n == 0:
            self.mean, self.m2, self.lo, self.hi = mean, m2, lo, hi
        else:
            delta = mean - self.mean
            total = self.n + n
            self.m2 += m2 + delta * delta * self.n * n / total
            self.mean += delta * n / total
            self.lo, self.hi = np.minimum(self.lo, lo), np.maximum(self.hi, hi)
        self.n += n
        self.frames += frames

    def result(self, media=False):
        def out(a):
            return np.asarray(a).reshape(-1, 1, 1).tolist() if media else np.asarray(a).tolist()
        return dict(min=out(self.lo), max=out(self.hi), mean=out(self.mean),
                    std=out(np.sqrt(np.maximum(self.m2 / self.n, 0))), count=[self.frames])


def table_stats(table):
    result = {}
    for key in table.column_names:
        a = values(table[key])
        m = Moments()
        m.add(a)
        result[key] = m.result()
    return result


def aggregate(rows):
    accum = {}
    for row in rows:
        for key, stat in row["stats"].items():
            n = stat["count"][0]
            mean = np.asarray(stat["mean"], dtype=np.float64)
            accum.setdefault(key, Moments()).merge(n, mean, np.square(stat["std"]) * n,
                np.asarray(stat["min"]), np.asarray(stat["max"]), n)
    return {key: value.result() for key, value in accum.items()}
