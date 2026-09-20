"""Visualize and measure how the mask branch guides YOLO26 predictions.

Edit the global settings below, then run:
    ./lib/bin/python tools/visualize_attention_map.py

The script compares the real mask against an all-zero counterfactual mask. It
saves P3/P4/P5 attention maps, overlays, detections, and numeric differences.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


# -----------------------------------------------------------------------------
# Global settings
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "runs/train/compact-attempt-1/weights/best.pt"
DATA_YAML = ROOT / "datasets/yolo/data.yaml"

# Leave these as None to use the first validation image and its matching mask.
IMAGE_PATH: str | Path | None = None
MASK_PATH: str | Path | None = None

OUTPUT_DIR = ROOT / "runs/attention/compact-attempt-1"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 640
CONFIDENCE = 0.25
IOU_THRESHOLD = 0.7
MAX_DETECTIONS = 100
OVERLAY_ALPHA = 0.45


sys.path.insert(0, str(ROOT))
from yolo26.config import load_data_config  # noqa: E402
from yolo26.dataloader import discover_images, letterbox, letterbox_mask, pair_masks, read_mask  # noqa: E402
from yolo26.model import YOLO26withMaskGuider, build_model  # noqa: E402


def resolve_inputs(data: dict) -> tuple[Path, Path]:
    image = Path(IMAGE_PATH).expanduser().resolve() if IMAGE_PATH else discover_images(data["val"])[0]
    if MASK_PATH:
        mask = Path(MASK_PATH).expanduser().resolve()
    else:
        mask_source = data.get("val_masks")
        if not mask_source:
            raise ValueError("DATA_YAML must define val_masks when MASK_PATH is None")
        mask = pair_masks([image], mask_source)[0]
    if not image.exists() or not mask.exists():
        raise FileNotFoundError(f"missing image or mask: {image}, {mask}")
    return image, mask


def load_model(device: torch.device):
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint must be a dictionary produced by train.py")
    route = checkpoint.get("model_route", "image")
    if route != "mask_guider":
        raise ValueError(f"checkpoint route is {route!r}, expected 'mask_guider'")
    nc = int(checkpoint["nc"])
    size = checkpoint.get("size", "n")
    mask_channels = int(checkpoint.get("mask_channels") or 1)
    model = build_model(size=size, nc=nc, model_route=route, mask_channels=mask_channels).to(device)
    model.load_compact(MODEL_PATH, strict=True)
    model.eval()
    names = checkpoint.get("names") or [str(i) for i in range(nc)]
    return model, names, mask_channels


def prepare_inputs(image_path: Path, mask_path: Path, mask_channels: int, device: torch.device):
    original = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if original is None:
        raise FileNotFoundError(f"cannot read image: {image_path}")
    h, w = original.shape[:2]
    empty_boxes = np.zeros((0, 4), dtype=np.float32)
    image, _, ratio_pad = letterbox(original, empty_boxes, IMAGE_SIZE)

    mask = read_mask(mask_path, mask_channels)
    if mask.shape[:2] != (h, w):
        raise ValueError(f"mask spatial shape {mask.shape[:2]} does not match image {(h, w)}")
    mask = letterbox_mask(mask, IMAGE_SIZE, ratio_pad)
    if mask.ndim == 2:
        mask = mask[..., None]

    image_tensor = torch.from_numpy(
        np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose(2, 0, 1))
    ).float().div_(255).unsqueeze(0).to(device)
    mask_tensor = torch.from_numpy(np.ascontiguousarray(mask.transpose(2, 0, 1))).float().unsqueeze(0).to(device)
    return original, image, mask, image_tensor, mask_tensor, ratio_pad


@torch.inference_mode()
def forward_with_attention(model: YOLO26withMaskGuider, image: torch.Tensor, mask: torch.Tensor):
    snapshots: dict[str, dict[str, torch.Tensor]] = {}
    handles = []
    levels = ("P3", "P4", "P5")

    for level, fusion in zip(levels, model.mask_fusion):
        def save_attention(_module, _inputs, output, level=level):
            snapshots.setdefault(level, {})["logits"] = output.detach().float().cpu()

        def save_fusion(_module, inputs, output, level=level):
            image_feature, mask_feature = inputs
            snapshots.setdefault(level, {}).update(
                image_feature=image_feature.detach().float().cpu(),
                mask_feature=mask_feature.detach().float().cpu(),
                fused_feature=output.detach().float().cpu(),
            )

        handles.append(fusion.attention.register_forward_hook(save_attention))
        handles.append(fusion.register_forward_hook(save_fusion))

    try:
        raw = model(image, mask)
    finally:
        for handle in handles:
            handle.remove()
    return raw, snapshots


def normalize_map(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(values.astype(np.float32))
    lo, hi = np.percentile(values, (1, 99))
    if hi <= lo:
        return np.zeros_like(values, dtype=np.uint8)
    return (np.clip((values - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def heatmap(values: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    values = cv2.resize(normalize_map(values), size, interpolation=cv2.INTER_CUBIC)
    return cv2.applyColorMap(values, cv2.COLORMAP_TURBO)


def overlay(base: np.ndarray, colored: np.ndarray) -> np.ndarray:
    return cv2.addWeighted(base, 1.0 - OVERLAY_ALPHA, colored, OVERLAY_ALPHA, 0)


def label_panel(panel: np.ndarray, text: str) -> np.ndarray:
    panel = panel.copy()
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(panel, text, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def draw_detections(image: np.ndarray, detections: torch.Tensor, names) -> np.ndarray:
    output = image.copy()
    for x1, y1, x2, y2, score, class_id in detections.cpu().tolist():
        class_id = int(class_id)
        color = tuple(int(x) for x in np.random.default_rng(class_id).integers(64, 255, 3))
        cv2.rectangle(output, (round(x1), round(y1)), (round(x2), round(y2)), color, 2)
        text = f"{names[class_id]} {score:.2f}"
        cv2.putText(output, text, (round(x1), max(15, round(y1) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return output


def raw_difference(real: dict, zero: dict) -> dict[str, float]:
    metrics = {}
    for branch in ("one2many", "one2one"):
        real_boxes, zero_boxes = real[branch]["boxes"].float(), zero[branch]["boxes"].float()
        real_scores = real[branch]["scores"].float().sigmoid()
        zero_scores = zero[branch]["scores"].float().sigmoid()
        metrics[f"{branch}_box_mae"] = float((real_boxes - zero_boxes).abs().mean())
        metrics[f"{branch}_probability_mae"] = float((real_scores - zero_scores).abs().mean())
        metrics[f"{branch}_probability_max_change"] = float((real_scores - zero_scores).abs().max())
    return metrics


def save_mask_channels(mask: np.ndarray, base: np.ndarray):
    panels = [label_panel(base, "Input image")]
    for index in range(mask.shape[-1]):
        colored = heatmap(mask[..., index], (base.shape[1], base.shape[0]))
        panels.append(label_panel(overlay(base, colored), f"Mask channel {index}"))
    cv2.imwrite(str(OUTPUT_DIR / "mask_channels.jpg"), np.concatenate(panels, axis=1))


def save_attention_figures(base: np.ndarray, real_maps: dict, zero_maps: dict) -> dict:
    metrics = {}
    summary_rows = []
    output_size = (base.shape[1], base.shape[0])
    for level in ("P3", "P4", "P5"):
        real_gate = real_maps[level]["logits"].sigmoid()[0, 0].numpy()
        zero_gate = zero_maps[level]["logits"].sigmoid()[0, 0].numpy()
        gate_delta = np.abs(real_gate - zero_gate)

        image_feature = real_maps[level]["image_feature"]
        fused_feature = real_maps[level]["fused_feature"]
        effect = (fused_feature - image_feature).abs().mean(1)[0].numpy()

        metrics[level] = {
            "gate_mean": float(real_gate.mean()),
            "gate_std": float(real_gate.std()),
            "gate_min": float(real_gate.min()),
            "gate_max": float(real_gate.max()),
            "real_vs_zero_gate_mae": float(gate_delta.mean()),
            "feature_absolute_effect_mean": float(effect.mean()),
        }

        real_heat = heatmap(real_gate, output_size)
        zero_heat = heatmap(zero_gate, output_size)
        delta_heat = heatmap(gate_delta, output_size)
        effect_heat = heatmap(effect, output_size)
        panels = [
            label_panel(real_heat, f"{level} gate (real mask)"),
            label_panel(overlay(base, real_heat), f"{level} overlay"),
            label_panel(zero_heat, f"{level} gate (zero mask)"),
            label_panel(delta_heat, f"{level} |real-zero|"),
            label_panel(effect_heat, f"{level} feature effect"),
        ]
        row = np.concatenate(panels, axis=1)
        cv2.imwrite(str(OUTPUT_DIR / f"attention_{level.lower()}.jpg"), row)
        summary_rows.append(row)
    cv2.imwrite(str(OUTPUT_DIR / "attention_summary.jpg"), np.concatenate(summary_rows, axis=0))
    return metrics


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(DEVICE)
    data = load_data_config(DATA_YAML)
    image_path, mask_path = resolve_inputs(data)
    model, names, mask_channels = load_model(device)
    original, prepared_image, mask, image_tensor, mask_tensor, ratio_pad = prepare_inputs(
        image_path, mask_path, mask_channels, device
    )

    real_raw, real_maps = forward_with_attention(model, image_tensor, mask_tensor)
    zero_raw, zero_maps = forward_with_attention(model, image_tensor, torch.zeros_like(mask_tensor))
    real_detections = model.head.postprocess(real_raw, CONFIDENCE, IOU_THRESHOLD, MAX_DETECTIONS, end2end=False)[0]
    zero_detections = model.head.postprocess(zero_raw, CONFIDENCE, IOU_THRESHOLD, MAX_DETECTIONS, end2end=False)[0]

    cv2.imwrite(str(OUTPUT_DIR / "input.jpg"), prepared_image)
    save_mask_channels(mask, prepared_image)
    attention_metrics = save_attention_figures(prepared_image, real_maps, zero_maps)
    cv2.imwrite(str(OUTPUT_DIR / "detections_real_mask.jpg"), draw_detections(prepared_image, real_detections, names))
    cv2.imwrite(str(OUTPUT_DIR / "detections_zero_mask.jpg"), draw_detections(prepared_image, zero_detections, names))

    metrics = {
        "model": str(MODEL_PATH),
        "image": str(image_path),
        "mask": str(mask_path),
        "mask_shape": list(mask_tensor.shape),
        "real_mask_detections": int(len(real_detections)),
        "zero_mask_detections": int(len(zero_detections)),
        "prediction_difference": raw_difference(real_raw, zero_raw),
        "attention": attention_metrics,
    }
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Saved attention visualizations to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
