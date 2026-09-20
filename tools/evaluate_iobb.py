"""Evaluate a YOLO26 checkpoint with prediction-normalized IoBB.

IoBB = intersection(prediction, ground truth) / area(prediction). This measures
whether a predicted region is contained by a coarse ground-truth box. It is an
auxiliary localization metric, not a replacement for IoU mAP.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "runs/train/compact-attempt-1/weights/best.pt"
DATA_YAML = ROOT / "datasets/yolo/data.yaml"
OUTPUT_PATH = ROOT / "runs/train/compact-attempt-1/iobb_metrics.json"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 640
BATCH_SIZE = 16
WORKERS = 8
CONFIDENCE = 0.001
NMS_IOU = 0.7
MAX_DETECTIONS = 300

sys.path.insert(0, str(ROOT))
from yolo26.config import load_data_config  # noqa: E402
from yolo26.dataloader import create_dataloader  # noqa: E402
from yolo26.losses import xywh2xyxy  # noqa: E402
from yolo26.metrics import compute_ap, match_predictions  # noqa: E402
from yolo26.model import build_model  # noqa: E402
from yolo26.utils import forward_batch, move_batch  # noqa: E402


def box_iobb(gt_boxes: torch.Tensor, pred_boxes: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Pairwise intersection divided by prediction area; shape is [GT, prediction]."""
    gt_left, gt_right = gt_boxes.float().unsqueeze(1).chunk(2, dim=2)
    pred_left, pred_right = pred_boxes.float().unsqueeze(0).chunk(2, dim=2)
    intersection = (torch.minimum(gt_right, pred_right) - torch.maximum(gt_left, pred_left)).clamp_(0).prod(2)
    prediction_area = (pred_right - pred_left).clamp_(0).prod(2)
    return intersection / (prediction_area + eps)


class IoBBMetrics:
    def __init__(self, nc: int, names: list[str]):
        self.nc = nc
        self.names = names
        self.thresholds = torch.linspace(0.5, 0.95, 10)
        self.stats = []

    @torch.no_grad()
    def update(self, predictions: list[torch.Tensor], batch: dict):
        for image_index, prediction in enumerate(predictions):
            target_mask = batch["batch_idx"] == image_index
            target_classes = batch["cls"][target_mask, 0].to(prediction.device)
            target_boxes = xywh2xyxy(batch["bboxes"][target_mask].to(prediction.device)) * IMAGE_SIZE
            if len(prediction) and len(target_boxes):
                correct = match_predictions(
                    prediction[:, 5], target_classes, box_iobb(target_boxes, prediction[:, :4]),
                    self.thresholds.to(prediction.device),
                )
            else:
                correct = torch.zeros((len(prediction), 10), dtype=torch.bool, device=prediction.device)
            self.stats.append(
                (correct.cpu(), prediction[:, 4].cpu(), prediction[:, 5].cpu(), target_classes.cpu())
            )

    def compute(self) -> dict:
        true_positive = torch.cat([item[0] for item in self.stats]).numpy()
        confidence = torch.cat([item[1] for item in self.stats]).numpy()
        predicted_class = torch.cat([item[2] for item in self.stats]).numpy()
        target_class = torch.cat([item[3] for item in self.stats]).numpy()
        order = np.argsort(-confidence)
        true_positive, predicted_class = true_positive[order], predicted_class[order]
        per_class, all_ap = {}, []

        for class_id in np.unique(target_class.astype(int)):
            predicted_mask = predicted_class == class_id
            target_count = int((target_class == class_id).sum())
            prediction_count = int(predicted_mask.sum())
            if prediction_count:
                false_positive = (1 - true_positive[predicted_mask]).cumsum(0)
                hits = true_positive[predicted_mask].cumsum(0)
                recall = hits / (target_count + 1e-16)
                precision = hits / (hits + false_positive + 1e-16)
                ap = np.asarray([compute_ap(recall[:, index], precision[:, index]) for index in range(10)])
            else:
                ap = np.zeros(10)
            all_ap.append(ap)
            per_class[self.names[class_id]] = {
                "targets": target_count,
                "predictions": prediction_count,
                "ap_iobb50": float(ap[0]),
                "ap_iobb50_95": float(ap.mean()),
            }

        mean_ap = np.stack(all_ap).mean(0)
        return {
            "metric": "IoBB = intersection / prediction_area",
            "thresholds": self.thresholds.tolist(),
            "map_iobb50": float(mean_ap[0]),
            "map_iobb50_95": float(mean_ap.mean()),
            "map_by_threshold": {f"{value:.2f}": float(ap) for value, ap in zip(self.thresholds, mean_ap)},
            "per_class": per_class,
        }


@torch.inference_mode()
def main():
    device = torch.device(DEVICE)
    data = load_data_config(DATA_YAML)
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    route = checkpoint.get("model_route", "image")
    mask_channels = int(checkpoint.get("mask_channels") or 1)
    model = build_model(
        size=checkpoint.get("size", "n"), nc=data["nc"], model_route=route, mask_channels=mask_channels
    ).to(device)
    model.load_compact(MODEL_PATH, strict=True)
    model.eval()
    loader = create_dataloader(
        data["val"], data["nc"], IMAGE_SIZE, BATCH_SIZE, WORKERS,
        mask_source=data.get("val_masks") if route == "mask_guider" else None,
        mask_channels=mask_channels, use_multiclass=False,
    )
    metrics = IoBBMetrics(data["nc"], data["names"])
    for batch in tqdm(loader, desc="IoBB validation"):
        batch = move_batch(batch, device)
        raw = forward_batch(model, batch)
        predictions = model.head.postprocess(
            raw, CONFIDENCE, NMS_IOU, MAX_DETECTIONS, end2end=False, multi_label=False
        )
        metrics.update(predictions, batch)

    result = metrics.compute()
    result.update(model=str(MODEL_PATH), images=len(loader.dataset))
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
