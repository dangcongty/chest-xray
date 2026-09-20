# SPDX-License-Identifier: AGPL-3.0-only
"""Image inference and visualization."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from dataloader import discover_images, letterbox
from model import YOLO26
from utils import select_device


def color(class_id):
    return tuple(int(x) for x in np.random.default_rng(class_id).integers(64, 255, 3))


def main():
    p = argparse.ArgumentParser(description="Predict with compact YOLO26")
    p.add_argument("--weights", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--output", default="runs/predict")
    p.add_argument("--size", choices=list("nsmxl"))
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--max-det", type=int, default=300)
    p.add_argument("--device", default="")
    p.add_argument("--nms", action="store_true", help="use one-to-many head and NMS instead of NMS-free head")
    args = p.parse_args()

    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    nc = int(ckpt.get("nc", 80))
    names = ckpt.get("names") or [str(i) for i in range(nc)]
    size = args.size or ckpt.get("size", "n")
    device = select_device(args.device)
    model = YOLO26(nc, size).to(device)
    model.load_compact(args.weights, strict=False)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    for path in discover_images(args.source):
        original = cv2.imread(str(path))
        if original is None:
            continue
        image, _, (ratio, (pad_x, pad_y)) = letterbox(original, np.zeros((0, 4), np.float32), args.imgsz)
        tensor = (
            torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).copy())
            .float()[None]
            .div_(255)
            .to(device)
        )
        pred = model.predict(tensor, args.conf, args.iou, args.max_det, end2end=not args.nms)[0].cpu()
        if len(pred):
            pred[:, [0, 2]] = (pred[:, [0, 2]] - pad_x) / ratio
            pred[:, [1, 3]] = (pred[:, [1, 3]] - pad_y) / ratio
            pred[:, [0, 2]].clamp_(0, original.shape[1])
            pred[:, [1, 3]].clamp_(0, original.shape[0])
        for x1, y1, x2, y2, score, cls in pred.tolist():
            c = color(int(cls))
            cv2.rectangle(original, (round(x1), round(y1)), (round(x2), round(y2)), c, 2)
            text = f"{names[int(cls)]} {score:.2f}"
            cv2.putText(
                original, text, (round(x1), max(round(y1) - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA
            )
        cv2.imwrite(str(output / path.name), original)
        print(f"{path}: {len(pred)} detections -> {output / path.name}")


if __name__ == "__main__":
    main()
