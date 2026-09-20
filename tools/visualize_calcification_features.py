"""Visualize whether Calcification evidence survives YOLO26 downsampling.

Runs the smallest, median, and largest Calcification boxes in the validation
split through the normal-label checkpoint. Outputs feature-energy maps,
class-specific Grad-CAM maps, and box-retention metrics for each scale.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "runs/train/compact-attempt-1/weights/best.pt"
DATA_YAML = ROOT / "datasets/yolo/data.yaml"
OUTPUT_DIR = ROOT / "runs/features/compact-attempt-1/calcification"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 640
CLASS_ID = 2
CLASS_NAME = "Calcification"
OVERLAY_ALPHA = 0.45

sys.path.insert(0, str(ROOT))
from yolo26.config import load_data_config  # noqa: E402
from yolo26.dataloader import (  # noqa: E402
    discover_images,
    image_to_label_path,
    letterbox,
    letterbox_mask,
    pair_masks,
    read_label,
    read_mask,
    xywhn_to_xyxy,
)
from yolo26.model import build_model  # noqa: E402


def select_cases(images: list[Path], nc: int) -> list[dict]:
    cases = []
    for image_path in images:
        labels = read_label(image_to_label_path(image_path), nc, use_multiclass=False)
        for row in labels[labels[:, 0] == CLASS_ID]:
            cases.append({"image": image_path, "label": row, "area": float(row[3] * row[4])})
    if not cases:
        raise RuntimeError(f"no class {CLASS_ID} labels found")
    cases.sort(key=lambda item: item["area"])
    indices = [0, len(cases) // 2, len(cases) - 1]
    names = ["smallest", "median", "largest"]
    return [{**cases[index], "case": name} for name, index in zip(names, indices)]


def load_model(device: torch.device):
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    route = checkpoint.get("model_route", "image")
    model = build_model(
        size=checkpoint.get("size", "n"),
        nc=int(checkpoint["nc"]),
        model_route=route,
        mask_channels=int(checkpoint.get("mask_channels") or 1),
    ).to(device)
    model.load_compact(MODEL_PATH, strict=True)
    model.eval()
    return model, checkpoint, route


def prepare(case: dict, data: dict, mask_channels: int, device: torch.device):
    original = cv2.imread(str(case["image"]), cv2.IMREAD_COLOR)
    if original is None:
        raise FileNotFoundError(case["image"])
    h, w = original.shape[:2]
    box = xywhn_to_xyxy(case["label"][None, 1:5], w, h)
    image, box, ratio_pad = letterbox(original, box, IMAGE_SIZE)
    image_tensor = torch.from_numpy(
        np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose(2, 0, 1))
    ).float().div_(255).unsqueeze(0).to(device)

    mask_tensor = None
    if mask_channels:
        mask_path = pair_masks([case["image"]], data["val_masks"])[0]
        mask = read_mask(mask_path, mask_channels)
        mask = letterbox_mask(mask, IMAGE_SIZE, ratio_pad)
        if mask.ndim == 2:
            mask = mask[..., None]
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask.transpose(2, 0, 1))).float().unsqueeze(0).to(device)
    return image, image_tensor, mask_tensor, box[0]


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(values.astype(np.float32))
    low, high = np.percentile(values, (1, 99))
    if high <= low:
        return np.zeros(values.shape, np.uint8)
    return (np.clip((values - low) / (high - low), 0, 1) * 255).astype(np.uint8)


def make_overlay(image: np.ndarray, values: np.ndarray, box: np.ndarray, title: str) -> np.ndarray:
    resized = cv2.resize(normalize(values), (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
    colored = cv2.applyColorMap(resized, cv2.COLORMAP_TURBO)
    panel = cv2.addWeighted(image, 1 - OVERLAY_ALPHA, colored, OVERLAY_ALPHA, 0)
    x1, y1, x2, y2 = np.round(box).astype(int)
    cv2.rectangle(panel, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(panel, title, (9, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def region_metrics(values: np.ndarray, box: np.ndarray) -> dict[str, float | bool | list[int]]:
    height, width = values.shape
    scale_x, scale_y = width / IMAGE_SIZE, height / IMAGE_SIZE
    x1 = max(0, min(width - 1, int(np.floor(box[0] * scale_x))))
    y1 = max(0, min(height - 1, int(np.floor(box[1] * scale_y))))
    x2 = max(x1 + 1, min(width, int(np.ceil(box[2] * scale_x))))
    y2 = max(y1 + 1, min(height, int(np.ceil(box[3] * scale_y))))
    inside = values[y1:y2, x1:x2]
    outside_mask = np.ones(values.shape, dtype=bool)
    outside_mask[y1:y2, x1:x2] = False
    outside = values[outside_mask]
    inside_mean = float(inside.mean())
    outside_mean = float(outside.mean()) if outside.size else 0.0
    total = float(values.clip(min=0).sum())
    peak = np.unravel_index(int(values.argmax()), values.shape)
    return {
        "feature_shape": [height, width],
        "box_cells": [x1, y1, x2, y2],
        "box_cell_width": x2 - x1,
        "box_cell_height": y2 - y1,
        "inside_mean": inside_mean,
        "outside_mean": outside_mean,
        "inside_outside_ratio": inside_mean / (outside_mean + 1e-12),
        "positive_mass_in_box": float(inside.clip(min=0).sum()) / (total + 1e-12),
        "peak_in_box": bool(y1 <= peak[0] < y2 and x1 <= peak[1] < x2),
    }


def analyze_case(model, route: str, image_tensor, mask_tensor, image, box, output_dir: Path):
    activations: dict[str, torch.Tensor] = {}
    handles = []
    modules = {
        "S2": model.model[0],
        "S4": model.model[1],
        "P3_backbone_S8": model.mask_fusion[0] if route == "mask_guider" else model.model[4],
        "P4_backbone_S16": model.mask_fusion[1] if route == "mask_guider" else model.model[6],
        "P5_backbone_S32": model.mask_fusion[2] if route == "mask_guider" else model.model[10],
        "P3_neck_S8": model.model[16],
        "P4_neck_S16": model.model[19],
        "P5_neck_S32": model.model[22],
    }

    for name, module in modules.items():
        def hook(_module, _inputs, output, name=name):
            activations[name] = output
            output.retain_grad()
        handles.append(module.register_forward_hook(hook))

    try:
        model.zero_grad(set_to_none=True)
        raw = model(image_tensor, mask_tensor) if route == "mask_guider" else model(image_tensor)
        class_probability = raw["one2many"]["scores"][:, CLASS_ID].sigmoid()
        best_probability, best_anchor = class_probability.max(dim=1)
        decoded_boxes, _ = model.head.decode(raw["one2many"])
        predicted_box = decoded_boxes[0, best_anchor.item()].detach()
        best_probability.sum().backward()
    finally:
        for handle in handles:
            handle.remove()

    energy_panels, cam_panels, metrics = [], [], {}
    for name, activation in activations.items():
        feature = activation.detach().float()[0]
        energy = feature.square().mean(0).sqrt().cpu().numpy()
        gradient = activation.grad.detach().float()[0]
        weights = gradient.mean((1, 2), keepdim=True)
        cam = torch.relu((weights * feature).sum(0)).cpu().numpy()
        energy_panels.append(make_overlay(image, energy, box, f"{name} feature energy"))
        cam_panels.append(make_overlay(image, cam, box, f"{name} Calcification Grad-CAM"))
        metrics[name] = {
            "energy": region_metrics(energy, box),
            "gradcam": region_metrics(cam, box),
        }

    cv2.imwrite(str(output_dir / "feature_energy.jpg"), np.concatenate(energy_panels, axis=1))
    cv2.imwrite(str(output_dir / "calcification_gradcam.jpg"), np.concatenate(cam_panels, axis=1))
    input_image = image.copy()
    x1, y1, x2, y2 = np.round(box).astype(int)
    cv2.rectangle(input_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite(str(output_dir / "input_box.jpg"), input_image)
    gt = torch.as_tensor(box, device=predicted_box.device, dtype=predicted_box.dtype)
    intersection = (
        (torch.minimum(predicted_box[2:], gt[2:]) - torch.maximum(predicted_box[:2], gt[:2]))
        .clamp(min=0)
        .prod()
    )
    union = (predicted_box[2:] - predicted_box[:2]).prod() + (gt[2:] - gt[:2]).prod() - intersection
    iou = intersection / union.clamp(min=1e-7)

    comparison = image.copy()
    px1, py1, px2, py2 = predicted_box.round().int().cpu().tolist()
    cv2.rectangle(comparison, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.rectangle(comparison, (px1, py1), (px2, py2), (0, 0, 255), 3)
    cv2.putText(
        comparison, "GT Calcification", (x1, max(24, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA,
    )
    cv2.putText(
        comparison,
        f"Pred {float(best_probability.detach()):.3f} IoU {float(iou):.3f}",
        (px1, max(24, py1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA,
    )
    cv2.imwrite(str(output_dir / "prediction_vs_gt.jpg"), comparison)
    return (
        metrics,
        float(best_probability.detach()),
        int(best_anchor.detach()),
        predicted_box.cpu().tolist(),
        float(iou.cpu()),
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(DEVICE)
    data = load_data_config(DATA_YAML)
    model, checkpoint, route = load_model(device)
    mask_channels = int(checkpoint.get("mask_channels") or 1) if route == "mask_guider" else 0
    cases = select_cases(discover_images(data["val"]), int(checkpoint["nc"]))
    report = {"model": str(MODEL_PATH), "class_id": CLASS_ID, "class_name": CLASS_NAME, "cases": {}}

    for case in cases:
        case_dir = OUTPUT_DIR / case["case"]
        case_dir.mkdir(parents=True, exist_ok=True)
        image, image_tensor, mask_tensor, box = prepare(case, data, mask_channels, device)
        metrics, probability, anchor, predicted_box, predicted_iou = analyze_case(
            model, route, image_tensor, mask_tensor, image, box, case_dir
        )
        report["cases"][case["case"]] = {
            "image": str(case["image"]),
            "normalized_box": case["label"][1:5].tolist(),
            "prepared_xyxy": box.tolist(),
            "prepared_width": float(box[2] - box[0]),
            "prepared_height": float(box[3] - box[1]),
            "best_raw_probability": probability,
            "best_anchor": anchor,
            "best_predicted_xyxy": predicted_box,
            "best_predicted_iou": predicted_iou,
            "layers": metrics,
        }

    path = OUTPUT_DIR / "metrics.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
