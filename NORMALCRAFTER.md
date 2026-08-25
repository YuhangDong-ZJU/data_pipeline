# NormalCrafter annotation worker

`annotate_normals_normalcrafter.py` discovers LeRobot videos named
`videos/chunk-*/observation.images.rgb_*/episode_*.mp4` and writes the matching
`observation.images.normal_*` MP4. Outputs are atomically published as
1280x720 H.264, CRF 17, `yuv420p`, at 15 FPS by default.

The worker loads NormalCrafter once, prefetches the next episode, decodes normal
latents in small chunks, records one JSON status per video, skips completed
outputs, and uses output locks for safe multi-machine operation. Normal vectors
remain in NormalCrafter's native view space; rotate them in the dataloader when
a robot-base convention is required.

## Existing server installation

On the 3539 server the launcher defaults to:

```text
NORMALCRAFTER_ROOT=/data2/normalcrafter_test/NormalCrafter
NORMALCRAFTER_PYTHON=/data2/normalcrafter_test/env/bin/python
HF_HOME=/data2/normalcrafter_test/hf_cache
```

Preview one camera and episode without loading the model:

```bash
bash run_normalcrafter.sh /data2/droid \
  --chunks chunk-000 \
  --cameras 01 \
  --episodes 0 \
  --dry-run
```

Process external cameras with four deterministic machine shards:

```bash
# Use shard indexes 0, 1, 2, and 3 on the four machines.
bash run_normalcrafter.sh /data2/droid \
  --cameras 01 02 \
  --num-shards 4 \
  --shard-index 0
```

All machines must use identical subset/chunk/camera/episode filters and write to
the same dataset or synchronized output root. A separate destination can be
selected with `--output-root`.

## Reproducing the dependency on another machine

NormalCrafter is kept outside this small data-pipeline repository because its
model source, environment, and Hugging Face checkpoints are independent
dependencies. Pin the tested upstream revision and apply the repository patch:

```bash
git clone https://github.com/Binyr/NormalCrafter.git
cd NormalCrafter
git checkout 75af9887a2cb14cd1ce3883c5773bc296565777c
git apply /path/to/data_pipeline/normalcrafter_long_video.patch
python3.10 -m venv /path/to/normalcrafter-env
/path/to/normalcrafter-env/bin/pip install -r requirements.txt
```

Then set `NORMALCRAFTER_ROOT` and `NORMALCRAFTER_PYTHON` before launching.

The default `--max-res 1024` runs a 1280x720 source at 1024x576 internally and
encodes the result at 1280x720. A direct 1280x720 run (`--max-res 1280`) exceeded
24 GB GPU memory in the original implementation, so it is not the production
default.
