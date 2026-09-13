"""Retry only a failed PNG read; keep failures distinct from fit rejection."""
import io
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError


class BadDepthImage(Exception):
    pass


def read_depth(path):
    path = Path(path)
    for attempt in range(2):
        # Filesystem errors (permissions, unavailable mount, etc.) are fatal.
        raw = path.read_bytes()
        try:
            with Image.open(io.BytesIO(raw)) as im:
                return np.asarray(im, dtype=np.float32)[::2, ::2] / 1000.
        except (UnidentifiedImageError, OSError, SyntaxError) as exc:
            if attempt:
                raise BadDepthImage(f'Depth decode failed after 2 reads: {path}: {exc}') from exc
            print(f'RETRY depth decode: {path}: {exc}', flush=True)


def excluded_candidate(job, failures):
    from .pointworld import POINTWORLD_COMMIT
    return dict(episode_index=job['episode_index'], source_episode_id=job['source']['source_episode_id'],
                source='pointworld_method_droid_initialization', pointworld_commit=POINTWORLD_COMMIT,
                excluded_bad_depth=True, failures=failures)
