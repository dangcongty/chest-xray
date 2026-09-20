# SPDX-License-Identifier: AGPL-3.0-only
"""Convert an official Ultralytics YOLO26 detection checkpoint to the compact format."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch

from model import YOLO26


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True, help="e.g. yolo26n.pt")
    p.add_argument("--output", help="default: <weights>.compact.pt")
    p.add_argument("--size", choices=list("nsmxl"))
    args = p.parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise SystemExit("Install the official converter dependency first: pip install ultralytics") from e

    source_wrapper = YOLO(args.weights)
    source = source_wrapper.model.float().cpu().eval()
    match = re.search(r"yolo26([nsmxl])", Path(args.weights).stem.lower())
    size = args.size or (match.group(1) if match else None)
    if not size:
        raise SystemExit("Cannot infer size; pass --size n|s|m|l|x")
    nc = int(source.model[-1].nc)
    target = YOLO26(nc, size)
    result = target.load_state_dict(source.state_dict(), strict=False)
    unexpected = [x for x in result.unexpected_keys if x != "model.23.stride"]
    missing = [x for x in result.missing_keys if x != "model.23.stride"]
    if missing or unexpected:
        raise RuntimeError(f"conversion mismatch: missing={missing}, unexpected={unexpected}")
    output = Path(args.output) if args.output else Path(args.weights).with_suffix(".compact.pt")
    torch.save(
        {
            "model": target.state_dict(),
            "ema": None,
            "size": size,
            "nc": nc,
            "names": getattr(source, "names", [str(i) for i in range(nc)]),
            "source": str(args.weights),
        },
        output,
    )
    print(f"Saved {output} ({target.num_parameters():,} parameters)")


if __name__ == "__main__":
    main()
