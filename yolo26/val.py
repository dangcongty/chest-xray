# SPDX-License-Identifier: AGPL-3.0-only
"""Validate compact YOLO26 checkpoints."""

from __future__ import annotations

import argparse
import json

import torch
from tqdm import tqdm

try:
    from .config import load_data_config
    from .dataloader import create_dataloader
    from .metrics import DetectionMetrics
    from .model import build_model
    from .utils import forward_batch, move_batch, select_device
except ImportError:
    from config import load_data_config
    from dataloader import create_dataloader
    from metrics import DetectionMetrics
    from model import build_model
    from utils import forward_batch, move_batch, select_device


@torch.inference_mode()
def evaluate(
    model, loader, device, names, image_size=640, conf=0.001, iou=0.7, max_det=300, end2end=False,
    criterion=None, use_multiclass=False,
):
    model.eval()
    metrics = DetectionMetrics(model.nc, names)
    loss_sum = torch.zeros(3, device=device)
    seen = 0
    for batch in tqdm(loader, desc="val", leave=False):
        batch = move_batch(batch, device)
        raw = forward_batch(model, batch)
        if criterion is not None:
            loss_vec, _ = criterion(raw, batch)
            loss_sum += loss_vec
        predictions = model.head.postprocess(raw, conf, iou, max_det, end2end, multi_label=use_multiclass)
        metrics.update(predictions, batch, image_size)
        seen += batch["img"].shape[0]
    result = metrics.compute()
    if criterion is not None:
        result["val_box_loss"], result["val_cls_loss"], result["val_l1_loss"] = (loss_sum / max(seen, 1)).tolist()
    return result


def main():
    parser = argparse.ArgumentParser(description="Validate YOLO26 compact")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--size", choices=list("nsmxl"), default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--end2end", action="store_true", help="validate NMS-free one-to-one head")
    parser.add_argument(
        "--use-multiclass", action=argparse.BooleanOptionalAction, default=None,
        help="read labels as `class1 class2 ... x y w h` (defaults to checkpoint train_args)",
    )
    args = parser.parse_args()

    data = load_data_config(args.data)
    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    size = args.size or (ckpt.get("size") if isinstance(ckpt, dict) else None) or "n"
    device = select_device(args.device)
    model_route = ckpt.get("model_route", "image") if isinstance(ckpt, dict) else "image"
    mask_channels = int(ckpt.get("mask_channels") or 1) if isinstance(ckpt, dict) else 1
    checkpoint_multiclass = (
        bool(ckpt.get("train_args", {}).get("use_multiclass", ckpt.get("use_multiclass", False)))
        if isinstance(ckpt, dict) else False
    )
    use_multiclass = checkpoint_multiclass if args.use_multiclass is None else args.use_multiclass
    model = build_model(size=size, nc=data["nc"], model_route=model_route, mask_channels=mask_channels).to(device)
    model.load_compact(args.weights, strict=False)
    loader = create_dataloader(
        data["val"], data["nc"], args.imgsz, args.batch, args.workers,
        mask_source=data.get("val_masks") if model_route == "mask_guider" else None,
        mask_channels=mask_channels,
        use_multiclass=use_multiclass,
    )
    result = evaluate(
        model, loader, device, data["names"], args.imgsz, args.conf, args.iou, args.max_det, args.end2end,
        use_multiclass=use_multiclass,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
