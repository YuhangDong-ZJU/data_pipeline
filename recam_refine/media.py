"""Decode every video frame; trim by presentation order, including B frames."""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import av
import numpy as np

from .common import require


def video_info(path):
    with av.open(str(path)) as container:
        require(len(container.streams.video) == 1 and not container.streams.audio, f"Unexpected video streams: {path}")
        s = container.streams.video[0]
        count = s.frames or sum(1 for _ in container.decode(s))
        return dict(frames=count, width=s.width, height=s.height,
                    fps=float(s.average_rate or 0), codec=s.codec_context.name,
                    pix_fmt=s.codec_context.format.name)


def decode_check(path, length, shape, fps, pixel_stats=False):
    from .stats import Moments
    stats = Moments() if pixel_stats else None
    count = 0
    first_time = None
    with av.open(str(path)) as container:
        require(len(container.streams.video) == 1 and not container.streams.audio, f"Unexpected streams: {path}")
        stream = container.streams.video[0]
        stream.thread_type = "SLICE"
        stream.codec_context.thread_count = 2
        require(abs(float(stream.average_rate or 0) - fps) < 1e-5, f"Wrong FPS: {path}")
        for frame in container.decode(stream):
            require(not frame.is_corrupt, f"Corrupt video frame: {path}:{count}")
            require((frame.height, frame.width) == tuple(shape[:2]), f"Wrong frame size: {path}")
            require(frame.pts is not None, f"Missing PTS: {path}:{count}")
            t = float(frame.pts * frame.time_base)
            if first_time is None:
                first_time = t
                require(abs(t) < 1e-4, f"Video does not start at zero: {path}")
            require(abs(t - count / fps) < 1e-4, f"Nonuniform/misaligned PTS: {path}:{count}")
            if stats is not None:
                rgb = frame.to_ndarray(format="rgb24").astype(np.float64) / 255
                stats.add(rgb.reshape(-1, 3), frames=1)
            count += 1
    require(count == length, f"Decoded {count}, expected {length}: {path}")
    return stats.result(media=True) if stats is not None else None


def trim_video(path, output, length, fps):
    """Lossless x264 from decoded YUV frames; no packet-count truncation.

    H.264 packet order differs from display order when B frames are present.
    Re-encoding with QP=0 preserves decoded source pixels and keeps exact PTS.
    """
    output = Path(output)
    with av.open(str(path)) as source, av.open(str(output), "w", format="mp4") as target:
        s = source.streams.video[0]
        enc = target.add_stream("libx264", rate=Fraction(str(fps)))
        enc.width, enc.height = s.width, s.height
        enc.pix_fmt = s.codec_context.format.name
        enc.options = {"crf": "0", "preset": "fast"}
        enc.codec_context.thread_count = 2
        tb = 1 / Fraction(str(fps))
        count = 0
        for frame in source.decode(s):
            if count == length:
                break
            require(not frame.is_corrupt, f"Corrupt source frame: {path}:{count}")
            frame.pts, frame.time_base = count, tb
            frame.pict_type = av.video.frame.PictureType.NONE
            for packet in enc.encode(frame):
                target.mux(packet)
            count += 1
        require(count == length, f"Source video is too short: {path}")
        for packet in enc.encode():
            target.mux(packet)
    # Final dataset validation decodes the result and checks every PTS/frame.


def assert_same_video_prefix(original, trimmed, length):
    with av.open(str(original)) as a, av.open(str(trimmed)) as b:
        aa, bb = a.decode(video=0), b.decode(video=0)
        for i in range(length):
            fa, fb = next(aa), next(bb)
            require(np.array_equal(fa.to_ndarray(), fb.to_ndarray()),
                    f"Lossless trim changed pixels: {trimmed}:{i}")
