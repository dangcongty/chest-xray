"""Build datasets/yolo_test/{images,labels,masks} from the annotated DICOM test set."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pydicom
from tqdm import tqdm


CLASS_NAMES = (
    "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
    "Consolidation", "ILD", "Infiltration", "Lung Opacity", "Nodule/Mass",
    "Other lesion", "Pleural effusion", "Pleural thickening", "Pneumothorax",
    "Pulmonary fibrosis",
)


def first_number(value) -> float:
    try:
        return float(value[0])
    except (TypeError, IndexError):
        return float(value)


def read_dicom(path: Path) -> np.ndarray:
    dataset = pydicom.dcmread(path)
    image = dataset.pixel_array.astype(np.float32)
    image = image * float(dataset.get("RescaleSlope", 1)) + float(dataset.get("RescaleIntercept", 0))
    center, width = dataset.get("WindowCenter"), dataset.get("WindowWidth")
    if center is not None and width is not None:
        center, width = first_number(center), first_number(width)
        low, high = center - width / 2, center + width / 2
    else:
        low, high = float(image.min()), float(image.max())
    if high <= low:
        raise ValueError(f"{path}: invalid DICOM pixel range [{low}, {high}]")
    image = np.clip((np.clip(image, low, high) - low) * (255.0 / (high - low)), 0, 255)
    if dataset.get("PhotometricInterpretation") == "MONOCHROME1":
        image = 255.0 - image
    return image.astype(np.uint8)


def letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    new_width, new_height = round(width * scale), round(height * scale)
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    left, top = (size - new_width) // 2, (size - new_height) // 2
    canvas = np.zeros((size, size), dtype=np.uint8)
    canvas[top : top + new_height, left : left + new_width] = resized
    return canvas, scale, left, top


def load_annotations(path: Path) -> tuple[dict[str, list[tuple[int, np.ndarray]]], dict[str, int]]:
    class_to_id = {name: index for index, name in enumerate(CLASS_NAMES)}
    annotations = defaultdict(list)
    ignored = defaultdict(int)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            name = row["class_name"].strip()
            if name not in class_to_id:
                ignored[name] += 1
                continue
            box = np.asarray([
                float(row["x_min"]), float(row["y_min"]),
                float(row["x_max"]), float(row["y_max"]),
            ], dtype=np.float32)
            annotations[row["image_id"].strip()].append((class_to_id[name], box))
    return annotations, dict(sorted(ignored.items()))


def yolo_lines(rows: list[tuple[int, np.ndarray]], scale: float, left: int, top: int,
               size: int) -> list[str]:
    lines = []
    for class_id, original in rows:
        box = original.copy()
        box[[0, 2]] = box[[0, 2]] * scale + left
        box[[1, 3]] = box[[1, 3]] * scale + top
        box = box.clip(0, size)
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            continue
        x, y = (x1 + x2) / (2 * size), (y1 + y2) / (2 * size)
        width, height = (x2 - x1) / size, (y2 - y1) / size
        lines.append(f"{class_id} {x:.6f} {y:.6f} {width:.6f} {height:.6f}")
    return lines


def place_mask(source: Path, destination: Path, mode: str) -> None:
    if mode == "copy":
        shutil.copy2(source, destination)
    elif mode == "hardlink":
        if destination.exists():
            destination.unlink()
        os.link(source, destination)
    else:
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        destination.symlink_to(source.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a YOLO test dataset from DICOM, CSV and masks")
    parser.add_argument("--dicom", type=Path, default=Path("datasets/dicom/test"))
    parser.add_argument("--annotations", type=Path, default=Path("datasets/dicom/annotations_test.csv"))
    parser.add_argument("--mask-source", type=Path, default=Path("datasets/dicom/test_masks"))
    parser.add_argument("--output", type=Path, default=Path("datasets/yolo_test"))
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--mask-mode", choices=("copy", "hardlink", "symlink"), default="copy")
    parser.add_argument("--allow-missing-masks", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dicom_paths = sorted(args.dicom.rglob("*.dicom"))
    if not dicom_paths:
        raise FileNotFoundError(f"No DICOM files found in {args.dicom}")
    masks = {path.stem: path for path in args.mask_source.rglob("*.npy")}
    missing_masks = [path.stem for path in dicom_paths if path.stem not in masks]
    if missing_masks and not args.allow_missing_masks:
        raise FileNotFoundError(
            f"Missing {len(missing_masks)}/{len(dicom_paths)} masks in {args.mask_source}. "
            "Generate them first with: python tools/get_segment.py --input datasets/dicom/test "
            "--output datasets/dicom/test_masks"
        )

    image_dir, label_dir, mask_dir = (
        args.output / "images", args.output / "labels", args.output / "masks"
    )
    for directory in (image_dir, label_dir, mask_dir):
        directory.mkdir(parents=True, exist_ok=True)
    annotations, ignored = load_annotations(args.annotations)
    converted = skipped = 0

    for dicom_path in tqdm(dicom_paths, desc="build yolo_test"):
        image_path = image_dir / f"{dicom_path.stem}.png"
        label_path = label_dir / f"{dicom_path.stem}.txt"
        mask_path = mask_dir / f"{dicom_path.stem}.npy"
        complete = image_path.exists() and label_path.exists() and (
            mask_path.exists() or dicom_path.stem not in masks
        )
        if complete and not args.overwrite:
            skipped += 1
            continue
        image, scale, left, top = letterbox(read_dicom(dicom_path), args.image_size)
        if not cv2.imwrite(str(image_path), image):
            raise RuntimeError(f"Could not write {image_path}")
        lines = yolo_lines(annotations.get(dicom_path.stem, []), scale, left, top, args.image_size)
        label_path.write_text("\n".join(lines), encoding="utf-8")
        if dicom_path.stem in masks:
            place_mask(masks[dicom_path.stem], mask_path, args.mask_mode)
        converted += 1

    print(f"Created: {converted} | skipped: {skipped} | images total: {len(dicom_paths)}")
    print(f"Output: {args.output.resolve()}")
    if ignored:
        print("Ignored classes not supported by the 14-class model:")
        print(", ".join(f"{name}={count}" for name, count in ignored.items()))


if __name__ == "__main__":
    main()
