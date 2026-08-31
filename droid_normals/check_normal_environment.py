#!/usr/bin/env python3
"""Validate the Python and CUDA runtime used by the normals workers."""

from __future__ import annotations

import argparse
import importlib
import re


def version_tuple(value: str) -> tuple[int, int]:
    match = re.match(r"(\d+)\.(\d+)", value)
    if match is None:
        raise RuntimeError(f"Cannot parse version: {value}")
    return int(match.group(1)), int(match.group(2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["legacy", "h100"], required=True)
    parser.add_argument(
        "--attention-backend",
        choices=["pytorch", "xformers"],
        default="pytorch",
    )
    parser.add_argument("--torch-only", action="store_true")
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False")
    if args.profile == "h100" and version_tuple(torch.__version__) < (2, 8):
        raise RuntimeError(
            f"H100 profile requires PyTorch >= 2.8, found {torch.__version__}"
        )

    device = torch.device("cuda", 0)
    capability = torch.cuda.get_device_capability(device)
    if args.profile == "h100" and capability < (9, 0):
        raise RuntimeError(
            f"H100 profile requires compute capability >= 9.0, found {capability}"
        )

    # A real CUDA kernel catches wheels that import but do not contain the GPU architecture.
    matrix = torch.ones((32, 32), device=device, dtype=torch.float16)
    result = matrix @ matrix
    if float(result[0, 0].item()) != 32.0:
        raise RuntimeError("CUDA kernel smoke test returned an unexpected result")

    if args.attention_backend == "xformers":
        import xformers
        import xformers.ops

        query = torch.randn((1, 16, 2, 32), device=device, dtype=torch.float16)
        xformers.ops.memory_efficient_attention(query, query, query)
        xformers_version = xformers.__version__
    else:
        xformers_version = "disabled"
    torch.cuda.synchronize(device)

    if not args.torch_only:
        for package in (
            "accelerate",
            "cv2",
            "decord",
            "diffusers",
            "hf_xet",
            "transformers",
        ):
            importlib.import_module(package)

    arches = ",".join(torch.cuda.get_arch_list())
    print(
        "Normal environment ready: "
        f"GPU={torch.cuda.get_device_name(device)}; capability={capability[0]}.{capability[1]}; "
        f"PyTorch={torch.__version__}; Torch CUDA={torch.version.cuda}; "
        f"xFormers={xformers_version}; arches={arches}",
        flush=True,
    )


if __name__ == "__main__":
    main()
