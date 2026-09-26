# SPDX-License-Identifier: AGPL-3.0-only
"""Evaluate a trained YOLO26 checkpoint on the annotated DICOM test set."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pydicom
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from yolo26.dataloader import read_mask, rasterize_guide, split_coarse_guide_boxes, xyxy_to_xywhn
from yolo26.metrics import DetectionMetrics
from yolo26.model import build_model, restore_stn_predictions
from yolo26.utils import forward_batch, move_batch, select_device

ROOT = Path(__file__).resolve().parent
DEFAULT_IMAGES = ROOT / "datasets/dicom/test"
DEFAULT_ANNOTATIONS = ROOT / "datasets/dicom/annotations_test.csv"
DEFAULT_WEIGHTS = ROOT / "runs/train/compact-attempt-1/weights/best.pt"
DEFAULT_MASKS = ROOT / "datasets/dicom/test_masks"
DEFAULT_OUTPUT = ROOT / "runs/test"


def first_number(value) -> float:
    """Convert a DICOM scalar or MultiValue to float."""
    try:
        return float(value[0])
    except (TypeError, IndexError):
        return float(value)


def read_dicom(path: Path) -> np.ndarray:
    """Decode a DICOM using the same windowing logic as tools/dicom2png.py."""
    ds = pydicom.dcmread(path)
    image = ds.pixel_array.astype(np.float32)
    image = image * float(ds.get("RescaleSlope", 1)) + float(ds.get("RescaleIntercept", 0))

    center, width = ds.get("WindowCenter"), ds.get("WindowWidth")
    if center is not None and width is not None:
        center, width = first_number(center), first_number(width)
        low, high = center - width / 2, center + width / 2
    else:
        low, high = float(image.min()), float(image.max())
    if high <= low:
        raise ValueError(f"{path}: invalid DICOM pixel range [{low}, {high}]")

    image = np.clip((np.clip(image, low, high) - low) * (255.0 / (high - low)), 0, 255)
    if ds.get("PhotometricInterpretation") == "MONOCHROME1":
        image = 255.0 - image
    return image.astype(np.uint8)


def load_annotations(
    path: Path, names: list[str]
) -> tuple[dict[str, list[tuple[int, list[float]]]], dict[str, int]]:
    name_to_id = {name: i for i, name in enumerate(names)}
    annotations: dict[str, list[tuple[int, list[float]]]] = defaultdict(list)
    ignored: dict[str, int] = defaultdict(int)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"image_id", "class_name", "x_min", "y_min", "x_max", "y_max"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns: {', '.join(sorted(missing))}")
        for line_number, row in enumerate(reader, 2):
            class_name = row["class_name"].strip()
            if class_name not in name_to_id:
                ignored[class_name] += 1
                continue
            try:
                box = [float(row[key]) for key in ("x_min", "y_min", "x_max", "y_max")]
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: invalid box coordinates") from exc
            annotations[row["image_id"].strip()].append((name_to_id[class_name], box))
    return annotations, dict(sorted(ignored.items()))


class DicomTestDataset(Dataset):
    def __init__(self, image_dir: Path, annotation_csv: Path, names: list[str], image_size: int,
                 coarse_guider: bool, guide_params: tuple[float, float, float, float, float],
                 mask_dir: Path | None = None, mask_channels: int = 4):
        self.paths = sorted(image_dir.rglob("*.dicom"))
        if not self.paths:
            raise FileNotFoundError(f"No .dicom files found in {image_dir}")
        self.names, self.nc, self.image_size = names, len(names), image_size
        self.annotations, self.ignored_annotations = load_annotations(annotation_csv, names)
        self.coarse_guider, self.guide_params = coarse_guider, guide_params
        self.mask_channels = mask_channels
        self.mask_paths = None
        if mask_dir is not None:
            candidates = {path.stem: path for path in mask_dir.rglob("*.npy")}
            missing_masks = [path.name for path in self.paths if path.stem not in candidates]
            if missing_masks:
                preview = ", ".join(missing_masks[:5])
                raise FileNotFoundError(
                    f"{len(missing_masks)} test masks are absent from {mask_dir}: {preview}. "
                    "Run: python tools/get_segment.py --input datasets/dicom/test "
                    "--output datasets/dicom/test_masks"
                )
            self.mask_paths = [candidates[path.stem] for path in self.paths]

        stems = {path.stem for path in self.paths}
        missing_images = sorted(set(self.annotations) - stems)
        if missing_images:
            preview = ", ".join(missing_images[:5])
            raise FileNotFoundError(f"{len(missing_images)} annotated images are absent from {image_dir}: {preview}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict:
        path = self.paths[index]
        gray = read_dicom(path)
        height, width = gray.shape[:2]
        scale = min(self.image_size / width, self.image_size / height)
        new_width, new_height = round(width * scale), round(height * scale)
        resized = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        left, top = (self.image_size - new_width) // 2, (self.image_size - new_height) // 2
        canvas = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        canvas[top : top + new_height, left : left + new_width] = resized

        rows = self.annotations.get(path.stem, [])
        classes = np.asarray([[row[0]] for row in rows], dtype=np.float32).reshape(-1, 1)
        boxes = np.asarray([row[1] for row in rows], dtype=np.float32).reshape(-1, 4)
        if len(boxes):
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + left
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + top
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, self.image_size)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, self.image_size)

        guide_classes = np.zeros((0, 1), dtype=np.float32)
        guide_boxes = np.zeros((0, 4), dtype=np.float32)
        if self.coarse_guider:
            classes, boxes, guide_classes, guide_boxes = split_coarse_guide_boxes(
                classes, boxes, self.image_size, *self.guide_params
            )

        normalized = xyxy_to_xywhn(boxes, self.image_size, self.image_size) if len(boxes) else boxes
        rgb = np.repeat(canvas[..., None], 3, axis=2)
        output = {
            "img": torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float().div_(255),
            "cls": torch.from_numpy(classes),
            "bboxes": torch.from_numpy(normalized),
            "path": str(path),
            "original_shape": (height, width),
            "ratio_pad": (scale, (left, top)),
        }
        if self.mask_paths is not None:
            mask = read_mask(self.mask_paths[index], self.mask_channels)
            if mask.shape[:2] != canvas.shape:
                raise ValueError(
                    f"Mask {self.mask_paths[index]} has shape {mask.shape[:2]}, expected {canvas.shape}"
                )
            output["mask"] = torch.from_numpy(mask.transpose(2, 0, 1).copy()).float()
        if self.coarse_guider:
            output["guide_mask"] = torch.from_numpy(
                rasterize_guide(guide_classes, guide_boxes, self.nc, self.image_size)
            )
        return output

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        output = {
            "img": torch.stack([sample["img"] for sample in batch]),
            "cls": torch.cat([sample["cls"] for sample in batch]),
            "bboxes": torch.cat([sample["bboxes"] for sample in batch]),
            "batch_idx": torch.cat([
                torch.full((len(sample["cls"]),), i, dtype=torch.long) for i, sample in enumerate(batch)
            ]),
        }
        if "guide_mask" in batch[0]:
            output["guide_mask"] = torch.stack([sample["guide_mask"] for sample in batch])
        if "mask" in batch[0]:
            output["mask"] = torch.stack([sample["mask"] for sample in batch])
        for key in ("path", "original_shape", "ratio_pad"):
            output[key] = [sample[key] for sample in batch]
        return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate YOLO26 on datasets/dicom/test")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--masks", type=Path, default=DEFAULT_MASKS,
                        help="Directory containing four-channel .npy masks for mask_guider")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="1")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--end2end", action="store_true", help="Use the NMS-free one-to-one head")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.imgsz % 32:
        raise ValueError("--imgsz must be divisible by 32")
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=False)
    names = checkpoint.get("names")
    if not names:
        raise ValueError(f"Checkpoint {args.weights} does not contain class names")
    if isinstance(names, dict):
        names = [names[key] for key in sorted(names, key=lambda value: int(value))]

    route = checkpoint.get("model_route", "image")
    mask_channels = int(checkpoint.get("mask_channels") or 1)
    if route == "mask_guider" and not args.masks.is_dir():
        raise FileNotFoundError(
            f"Mask directory not found: {args.masks}. Generate it with: "
            "python tools/get_segment.py --input datasets/dicom/test --output datasets/dicom/test_masks"
        )
    train_args = checkpoint.get("train_args", {})
    guide_params = tuple(float(train_args.get(key, default)) for key, default in (
        ("guide_iobb", 0.8), ("guide_min_area_ratio", 4.0), ("guide_min_area", 0.10),
        ("guide_min_width", 0.35), ("guide_min_height", 0.50),
    ))
    dataset = DicomTestDataset(
        args.images, args.annotations, names, args.imgsz, route == "coarse_guider", guide_params,
        mask_dir=args.masks if route == "mask_guider" else None, mask_channels=mask_channels,
    )
    if dataset.ignored_annotations:
        details = ", ".join(f"{name}={count}" for name, count in dataset.ignored_annotations.items())
        print(f"Ignoring annotations for classes absent from the checkpoint: {details}")
    loader = DataLoader(
        dataset, batch_size=args.batch, shuffle=False, num_workers=args.workers,
        pin_memory=torch.cuda.is_available(), persistent_workers=args.workers > 0,
        collate_fn=dataset.collate_fn,
    )

    device = select_device(args.device)
    model = build_model(
        size=checkpoint.get("size", "n"), nc=len(names), model_route=route,
        mask_channels=mask_channels,
    ).to(device)
    model.load_compact(args.weights, strict=False)
    model.eval()
    metrics = DetectionMetrics(len(names), names)
    prediction_rows = []

    for batch in tqdm(loader, desc="test"):
        device_batch = move_batch(batch, device)
        raw = forward_batch(model, device_batch)
        predictions = model.head.postprocess(
            raw, args.conf, args.iou, args.max_det, args.end2end,
            multi_label=bool(
                checkpoint.get("train_args", {}).get(
                    "use_multiclass", checkpoint.get("use_multiclass", False)
                )
            ),
        )
        if "stn_theta" in raw:
            predictions = restore_stn_predictions(predictions, raw["stn_theta"], device_batch["img"].shape[-2:])
        metrics.update(predictions, device_batch, args.imgsz)
        for prediction, path, shape, ratio_pad in zip(
            predictions, batch["path"], batch["original_shape"], batch["ratio_pad"]
        ):
            prediction = prediction.detach().cpu().clone()
            scale, (left, top) = ratio_pad
            height, width = shape
            if len(prediction):
                prediction[:, [0, 2]] = (prediction[:, [0, 2]] - left) / scale
                prediction[:, [1, 3]] = (prediction[:, [1, 3]] - top) / scale
                prediction[:, [0, 2]].clamp_(0, width)
                prediction[:, [1, 3]].clamp_(0, height)
            for x1, y1, x2, y2, score, class_id in prediction.tolist():
                prediction_rows.append({
                    "image_id": Path(path).stem, "class_name": names[int(class_id)],
                    "class_id": int(class_id), "confidence": score,
                    "x_min": x1, "y_min": y1, "x_max": x2, "y_max": y2,
                })

    args.output.mkdir(parents=True, exist_ok=True)
    result = metrics.compute()
    result.update({
        "weights": str(args.weights.resolve()), "images": len(dataset),
        "annotations": sum(len(rows) for rows in dataset.annotations.values()),
        "ignored_annotations": dataset.ignored_annotations,
        "model_route": route,
    })
    metrics_path = args.output / "metrics.json"
    predictions_path = args.output / "predictions.csv"
    per_class_path = args.output / "per_class_metrics.csv"
    metrics_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    fields = ["image_id", "class_name", "class_id", "confidence", "x_min", "y_min", "x_max", "y_max"]
    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(prediction_rows)
    per_class_fields = ["class_name", "targets", "precision", "recall", "map50", "map50_95"]
    with per_class_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_class_fields)
        writer.writeheader()
        for class_name in names:
            values = result["per_class"].get(class_name, {})
            writer.writerow({
                "class_name": class_name,
                "targets": values.get("images_targets", 0),
                "precision": values.get("precision", 0.0),
                "recall": values.get("recall", 0.0),
                "map50": values.get("map50", 0.0),
                "map50_95": values.get("map50_95", 0.0),
            })

    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"All classes: mAP50={result['map50']:.6f}, mAP50-95={result['map50_95']:.6f}")
    print(f"Metrics: {metrics_path}")
    print(f"Per-class metrics: {per_class_path}")
    print(f"Predictions: {predictions_path}")


if __name__ == "__main__":
    main()
