# SPDX-License-Identifier: AGPL-3.0-only
"""Detection IoU, precision, recall and COCO-style mAP."""

from __future__ import annotations

import numpy as np
import torch

try:
    from .losses import xywh2xyxy
except ImportError:
    from losses import xywh2xyxy


def box_iou(box1: torch.Tensor, box2: torch.Tensor, eps=1e-7):
    (a1, a2), (b1, b2) = box1.float().unsqueeze(1).chunk(2, 2), box2.float().unsqueeze(0).chunk(2, 2)
    inter = (torch.minimum(a2, b2) - torch.maximum(a1, b1)).clamp_(0).prod(2)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


def match_predictions(pred_cls, true_cls, iou, thresholds):
    """Greedily make one-to-one matches at each IoU threshold."""
    correct = torch.zeros((pred_cls.shape[0], thresholds.numel()), dtype=torch.bool, device=pred_cls.device)
    class_mask = true_cls[:, None] == pred_cls[None]
    for j, threshold in enumerate(thresholds):
        candidates = torch.where((iou >= threshold) & class_mask)
        if candidates[0].numel() == 0:
            continue
        pairs = torch.stack(candidates, 1)
        values = iou[candidates]
        pairs = pairs[values.argsort(descending=True)]
        used_gt, used_pred = set(), set()
        for gt_i, pred_i in pairs.tolist():
            if gt_i not in used_gt and pred_i not in used_pred:
                used_gt.add(gt_i)
                used_pred.add(pred_i)
                correct[pred_i, j] = True
    return correct


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return float(np.trapezoid(np.interp(x, mrec, mpre), x))


class DetectionMetrics:
    def __init__(self, nc: int, names=None):
        self.nc = nc
        self.names = names or [str(i) for i in range(nc)]
        self.iouv = torch.linspace(0.5, 0.95, 10)
        self.stats: list[tuple[torch.Tensor, ...]] = []

    @torch.no_grad()
    def update(self, predictions: list[torch.Tensor], batch: dict, image_size: int):
        batch_idx = batch["batch_idx"]
        for i, pred in enumerate(predictions):
            mask = batch_idx == i
            image_cls = batch["cls"][mask]
            image_boxes = batch["bboxes"][mask]
            if image_cls.shape[1] > 1:
                box_idx, true_cls = torch.where(image_cls > 0.5)
                true_boxes = image_boxes[box_idx].to(pred.device)
                true_cls = true_cls.to(pred.device)
            else:
                true_cls = image_cls[:, 0].to(pred.device)
                true_boxes = image_boxes.to(pred.device)
            if len(true_boxes):
                true_boxes = xywh2xyxy(true_boxes) * image_size
            if len(pred) and len(true_boxes):
                correct = match_predictions(
                    pred[:, 5], true_cls, box_iou(true_boxes, pred[:, :4]), self.iouv.to(pred.device)
                )
            else:
                correct = torch.zeros((len(pred), 10), dtype=torch.bool, device=pred.device)
            self.stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), true_cls.cpu()))

    def compute(self) -> dict[str, float | dict]:
        if not self.stats:
            return {"precision": 0.0, "recall": 0.0, "map50": 0.0, "map50_95": 0.0, "per_class": {}}
        tp = torch.cat([x[0] for x in self.stats]).numpy()
        conf = torch.cat([x[1] for x in self.stats]).numpy()
        pred_cls = torch.cat([x[2] for x in self.stats]).numpy()
        target_cls = torch.cat([x[3] for x in self.stats]).numpy()
        order = np.argsort(-conf)
        tp, conf, pred_cls = tp[order], conf[order], pred_cls[order]
        classes = np.unique(target_cls.astype(int))
        aps, best_p, best_r, per_class = [], [], [], {}
        for c in classes:
            pred_mask = pred_cls == c
            n_gt, n_pred = int((target_cls == c).sum()), int(pred_mask.sum())
            if n_gt == 0:
                continue
            if n_pred == 0:
                ap = np.zeros(10)
                p = r = 0.0
            else:
                false_pos = (1 - tp[pred_mask]).cumsum(0)
                true_pos = tp[pred_mask].cumsum(0)
                recall = true_pos / (n_gt + 1e-16)
                precision = true_pos / (true_pos + false_pos + 1e-16)
                ap = np.array([compute_ap(recall[:, j], precision[:, j]) for j in range(10)])
                f1 = 2 * precision[:, 0] * recall[:, 0] / (precision[:, 0] + recall[:, 0] + 1e-16)
                best = int(f1.argmax())
                p, r = float(precision[best, 0]), float(recall[best, 0])
            aps.append(ap)
            best_p.append(p)
            best_r.append(r)
            per_class[self.names[c] if c < len(self.names) else str(c)] = {
                "images_targets": n_gt,
                "precision": p,
                "recall": r,
                "map50": float(ap[0]),
                "map50_95": float(ap.mean()),
            }
        if not aps:
            return {"precision": 0.0, "recall": 0.0, "map50": 0.0, "map50_95": 0.0, "per_class": {}}
        aps = np.stack(aps)
        return {
            "precision": float(np.mean(best_p)),
            "recall": float(np.mean(best_r)),
            "map50": float(aps[:, 0].mean()),
            "map50_95": float(aps.mean()),
            "per_class": per_class,
        }

    def reset(self):
        self.stats.clear()
