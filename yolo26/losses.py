# SPDX-License-Identifier: AGPL-3.0-only
"""YOLO26 detection loss: TAL/STAL assignment, CIoU, BCE and DFL-free L1."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

try:
    from .model import dist2bbox, make_anchors
except ImportError:
    from model import dist2bbox, make_anchors


def xywh2xyxy(x):
    y = x.clone()
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def xyxy2xywh(x):
    y = x.clone()
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2
    y[..., 2] = x[..., 2] - x[..., 0]
    y[..., 3] = x[..., 3] - x[..., 1]
    return y


def bbox2dist(anchor_points, bbox):
    x1y1, x2y2 = bbox.chunk(2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1)


def bbox_iou(box1, box2, eps=1e-7, ciou=False):
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp_(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union
    if not ciou:
        return iou
    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)) / 4
    v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    return iou - (rho2 / (cw.pow(2) + ch.pow(2) + eps) + v * alpha)


class TaskAlignedAssigner(nn.Module):
    """Task-aligned assigner with YOLO26 small-target-aware candidate expansion."""

    def __init__(
        self, topk, num_classes, alpha=0.5, beta=6.0, strides=(8, 16, 32), topk2=None,
        eps=1e-9, use_multiclass=False,
    ):
        super().__init__()
        self.topk, self.topk2 = topk, topk2 or topk
        self.num_classes, self.alpha, self.beta, self.eps = num_classes, alpha, beta, eps
        self.use_multiclass = use_multiclass
        self.stride_val = strides[1] if len(strides) > 1 else strides[0]

    @torch.no_grad()
    def forward(self, scores, boxes, points, gt_labels, gt_boxes, mask_gt):
        bs, na, _ = scores.shape
        n_gt = gt_boxes.shape[1]
        if n_gt == 0:
            return (
                torch.full_like(scores[..., 0], self.num_classes),
                torch.zeros_like(boxes),
                torch.zeros_like(scores),
                torch.zeros_like(scores[..., 0], dtype=torch.bool),
                torch.zeros_like(scores[..., 0], dtype=torch.long),
            )
        inside = self._inside(points, gt_boxes, mask_gt)
        shape = (bs, n_gt, na)
        indices = inside.nonzero(as_tuple=True)
        overlaps = boxes.new_zeros(shape)
        align = scores.new_zeros(shape)
        if indices[0].numel():
            if self.use_multiclass:
                candidate_scores = scores[indices[0], indices[2]]
                candidate_labels = gt_labels[indices[0], indices[1]].bool()
                cls_score = candidate_scores.masked_fill(~candidate_labels, 0).amax(-1)
            else:
                cls_score = scores[indices[0], indices[2], gt_labels[indices[0], indices[1], 0].long()]
            ious = bbox_iou(gt_boxes[indices[:2]], boxes[indices[0], indices[2]], ciou=True).squeeze(-1).clamp_(0)
            ious = ious.to(overlaps.dtype)
            overlaps[indices] = ious
            alignment = cls_score.pow(self.alpha) * ious.pow(self.beta)
            align[indices] = alignment.to(align.dtype)

        k = min(self.topk, na)
        top_values, top_indices = torch.topk(align, k, dim=-1)
        valid = mask_gt.expand(-1, -1, k).bool() & (top_values.amax(-1, keepdim=True) > self.eps)
        top_indices.masked_fill_(~valid, 0)
        mask_pos = torch.zeros(shape, dtype=torch.int8, device=scores.device)
        mask_pos.scatter_add_(-1, top_indices, torch.ones_like(top_indices, dtype=torch.int8))
        mask_pos.masked_fill_(mask_pos > 1, 0)
        mask_pos = mask_pos.bool() & inside & mask_gt.bool()

        fg_count = mask_pos.sum(-2)
        multi = (fg_count.unsqueeze(1) > 1).expand_as(mask_pos)
        best_gt = overlaps.argmax(1)
        best_mask = torch.zeros_like(mask_pos).scatter_(1, best_gt[:, None], True)
        mask_pos = torch.where(multi, best_mask, mask_pos)

        if self.topk2 != self.topk:
            k2 = min(self.topk2, na)
            selected = align * mask_pos
            indices2 = torch.topk(selected, k2, dim=-1).indices
            reduced = torch.zeros_like(mask_pos).scatter_(-1, indices2, True)
            mask_pos &= reduced

        fg_mask = mask_pos.sum(-2).bool()
        target_gt_idx = mask_pos.long().argmax(-2)
        batch_idx = torch.arange(bs, device=scores.device)[:, None]
        target_boxes = gt_boxes[batch_idx, target_gt_idx]
        if self.use_multiclass:
            target_scores = gt_labels[batch_idx, target_gt_idx].to(scores.dtype)
            target_labels = target_scores.argmax(-1)
        else:
            target_labels = gt_labels.long().squeeze(-1)[batch_idx, target_gt_idx].clamp_(0)
            target_scores = torch.zeros((bs, na, self.num_classes), dtype=scores.dtype, device=scores.device)
            target_scores.scatter_(2, target_labels[..., None], 1)
        target_scores *= fg_mask[..., None]

        align *= mask_pos
        pos_align = align.amax(-1, keepdim=True)
        overlaps *= mask_pos
        pos_iou = overlaps.amax(-1, keepdim=True)
        norm = (align * pos_iou / (pos_align + self.eps)).amax(-2).unsqueeze(-1)
        target_scores *= norm
        return target_labels, target_boxes, target_scores, fg_mask, target_gt_idx

    def _inside(self, points, gt_boxes, mask_gt, eps=1e-9):
        boxes_xywh = xyxy2xywh(gt_boxes)
        tiny = boxes_xywh[..., 2:] < self.stride_val
        boxes_xywh[..., 2:] = torch.where(
            (tiny * mask_gt).bool(),
            torch.tensor(self.stride_val, dtype=boxes_xywh.dtype, device=boxes_xywh.device),
            boxes_xywh[..., 2:],
        )
        boxes = xywh2xyxy(boxes_xywh)
        lt, rb = boxes.unsqueeze(2).chunk(2, -1)
        return (
            (points[:, 0] - lt[..., 0] > eps)
            & (points[:, 1] - lt[..., 1] > eps)
            & (rb[..., 0] - points[:, 0] > eps)
            & (rb[..., 1] - points[:, 1] > eps)
        )


class DetectionLoss:
    def __init__(self, model, box=7.5, cls=0.5, l1=1.5, topk=10, topk2=None, use_multiclass=False):
        self.device = next(model.parameters()).device
        self.head = model.head
        self.nc = model.nc
        self.box_gain, self.cls_gain, self.l1_gain = box, cls, l1
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.use_multiclass = use_multiclass
        self.assigner = TaskAlignedAssigner(
            topk, self.nc, 0.5, 6.0, self.head.stride.tolist(), topk2,
            use_multiclass=use_multiclass,
        )

    def _preprocess(self, batch, batch_size, image_hw):
        targets = torch.cat((batch["batch_idx"][:, None], batch["cls"], batch["bboxes"]), 1).to(self.device)
        target_width = self.nc + 4 if self.use_multiclass else 5
        if targets.shape[0] == 0:
            return torch.zeros(batch_size, 0, target_width, device=self.device)
        batch_idx = targets[:, 0].long()
        counts = torch.bincount(batch_idx, minlength=batch_size)
        out = torch.zeros(batch_size, int(counts.max()), target_width, device=self.device)
        offsets = counts.cumsum(0) - counts
        within = torch.arange(len(targets), device=self.device) - offsets[batch_idx]
        out[batch_idx, within] = targets[:, 1:]
        scale = torch.tensor([image_hw[1], image_hw[0], image_hw[1], image_hw[0]], device=self.device)
        out[..., -4:] = xywh2xyxy(out[..., -4:] * scale)
        return out

    def __call__(self, pred, batch):
        pred_dist = pred["boxes"].permute(0, 2, 1).contiguous()
        pred_scores = pred["scores"].permute(0, 2, 1).contiguous()
        points, stride_tensor = make_anchors(pred["feats"], self.head.stride)
        bs = pred_scores.shape[0]
        image_hw = (
            pred["feats"][0].shape[2] * int(self.head.stride[0]),
            pred["feats"][0].shape[3] * int(self.head.stride[0]),
        )
        targets = self._preprocess(batch, bs, image_hw)
        if self.use_multiclass:
            gt_labels, gt_boxes = targets.split((self.nc, 4), 2)
        else:
            gt_labels, gt_boxes = targets.split((1, 4), 2)
        mask_gt = gt_boxes.sum(2, keepdim=True).gt_(0)
        pred_boxes = dist2bbox(pred_dist, points.unsqueeze(0))
        # Assignment must stay in FP32: TAL uses eps=1e-9, which underflows to
        # zero in FP16 and turns the normalization into 0/0 (NaN).
        _, target_boxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().float().sigmoid(),
            pred_boxes.detach().float() * stride_tensor.float(),
            points.float() * stride_tensor.float(),
            gt_labels,
            gt_boxes,
            mask_gt,
        )
        score_sum = target_scores.sum().clamp_(min=1)
        cls_loss = self.bce(pred_scores, target_scores.to(pred_scores.dtype)).sum() / score_sum
        fg_idx = fg_mask.nonzero(as_tuple=True)
        weight = target_scores[fg_idx].sum(-1, keepdim=True)
        iou = bbox_iou(pred_boxes[fg_idx], (target_boxes / stride_tensor)[fg_idx], ciou=True)
        box_loss = ((1 - iou) * weight).sum() / score_sum
        target_ltrb = bbox2dist(points, target_boxes / stride_tensor)
        image_size = pred_dist.new_tensor([image_hw[1], image_hw[0], image_hw[1], image_hw[0]])
        strides4 = stride_tensor.repeat(1, 4)
        pred_norm = pred_dist * strides4 / image_size
        target_norm = target_ltrb * strides4 / image_size
        l1_loss = (
            F.l1_loss(pred_norm[fg_idx], target_norm[fg_idx], reduction="none").mean(-1, keepdim=True) * weight
        ).sum() / score_sum
        components = torch.stack((box_loss * self.box_gain, cls_loss * self.cls_gain, l1_loss * self.l1_gain))
        return components * bs, {
            "box": components[0].detach(),
            "cls": components[1].detach(),
            "l1": components[2].detach(),
        }


class YOLO26Loss:
    """Progressive dual-head loss; the one-to-many weight decays toward 0.1."""

    def __init__(
        self, model, epochs=100, box=7.5, cls=0.5, l1=1.5, use_multiclass=False,
        guide_loss_weight=0.5,
    ):
        self.one2many = DetectionLoss(model, box, cls, l1, topk=10, use_multiclass=use_multiclass)
        self.one2one = DetectionLoss(model, box, cls, l1, topk=7, topk2=1, use_multiclass=use_multiclass)
        self.epochs, self.epoch = epochs, 0
        self.guide_loss_weight = guide_loss_weight

    @property
    def o2m_weight(self):
        return max(1 - self.epoch / max(self.epochs - 1, 1), 0) * 0.7 + 0.1

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __call__(self, preds, batch):
        o2m, _ = self.one2many(preds["one2many"], batch)
        o2o, items = self.one2one(preds["one2one"], batch)
        w = self.o2m_weight
        loss = o2m * w + o2o * (1 - w)
        output_items = {**items, "o2m_weight": w}
        if "guide_logits" in preds:
            if "guide_mask" not in batch:
                raise KeyError("coarse-guider predictions require batch['guide_mask']")
            target = batch["guide_mask"].float()
            scale_losses = []
            for logits in preds["guide_logits"]:
                resized = F.interpolate(target, size=logits.shape[-2:], mode="nearest")
                positives = resized.sum()
                negatives = resized.numel() - positives
                pos_weight = (negatives / positives.clamp_min(1)).clamp(1, 20).detach()
                bce = F.binary_cross_entropy_with_logits(logits.float(), resized, pos_weight=pos_weight)
                probabilities = logits.float().sigmoid()
                intersection = (probabilities * resized).sum(dim=(2, 3))
                dice = 1 - ((2 * intersection + 1) / (
                    probabilities.sum(dim=(2, 3)) + resized.sum(dim=(2, 3)) + 1
                )).mean()
                scale_losses.append(bce + dice)
            # DetectionLoss returns batch-scaled components; keep the auxiliary
            # component on the same scale so batch size does not alter its ratio.
            guide_loss = torch.stack(scale_losses).mean() * self.guide_loss_weight * target.shape[0]
            loss = torch.cat((loss, guide_loss.unsqueeze(0)))
            output_items["guide"] = guide_loss.detach()
        return loss, output_items
