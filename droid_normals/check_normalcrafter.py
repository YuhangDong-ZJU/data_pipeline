#!/usr/bin/env python3
"""Load the pinned NormalCrafter pipeline as an environment/GPU smoke test."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--normalcrafter-root", type=Path, required=True)
    parser.add_argument("--unet-path", default="Yanrui95/NormalCrafter")
    parser.add_argument(
        "--pretrain-path",
        default="stabilityai/stable-video-diffusion-img2vid-xt",
    )
    parser.add_argument("--cpu-offload", choices=["none", "model", "sequential"], default="none")
    parser.add_argument("--verbose-inference", action="store_true")
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location("droid_normals_worker", args.worker)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import worker: {args.worker}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False")
    runner = module.NormalCrafterRunner(
        SimpleNamespace(
            normalcrafter_root=args.normalcrafter_root,
            unet_path=args.unet_path,
            pretrain_path=args.pretrain_path,
            cpu_offload=args.cpu_offload,
            verbose_inference=args.verbose_inference,
        )
    )
    print(
        f"NormalCrafter ready: {torch.cuda.get_device_name(0)}, "
        f"PyTorch {torch.__version__}, model load {runner.load_seconds:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
