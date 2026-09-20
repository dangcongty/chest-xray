"""Build a multi-class YOLO-like dataset with one row per merged box.

Output rows use the variable-width format ``c1 c2 ... x y w h``. The last four
values are always normalized box geometry; every preceding value is a class ID.

Run:
    ./lib/bin/python make_data_multi_classes.py
"""

from __future__ import annotations

import os
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import yaml


# -----------------------------------------------------------------------------
# Global settings
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = ROOT / "datasets/yolo"
OUTPUT_ROOT = ROOT / "datasets/yolo_multi_class"
IOU_THRESHOLD = 0.80

# Hardlinks do not duplicate image data. The script automatically falls back to
# copying when source/output are on different filesystems.
IMAGE_MODE = "hardlink"  # hardlink | copy | symlink
OVERWRITE = True
DECIMALS = 6

IMAGE_SUFFIXES = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    output = boxes.copy()
    output[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    output[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    output[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    output[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return output


def box_iou(boxes: np.ndarray) -> np.ndarray:
    """Return the pairwise IoU matrix for normalized xywh boxes."""
    xyxy = xywh_to_xyxy(boxes)
    top_left = np.maximum(xyxy[:, None, :2], xyxy[None, :, :2])
    bottom_right = np.minimum(xyxy[:, None, 2:], xyxy[None, :, 2:])
    intersection_wh = np.clip(bottom_right - top_left, 0, None)
    intersection = intersection_wh[..., 0] * intersection_wh[..., 1]
    area = np.clip(xyxy[:, 2] - xyxy[:, 0], 0, None) * np.clip(xyxy[:, 3] - xyxy[:, 1], 0, None)
    union = area[:, None] + area[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def connected_groups(boxes: np.ndarray, threshold: float) -> list[list[int]]:
    """Group boxes by transitive IoU connections above the threshold."""
    count = len(boxes)
    if count == 0:
        return []
    parent = list(range(count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a: int, b: int):
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    ious = box_iou(boxes)
    rows, cols = np.where(np.triu(ious > threshold, k=1))
    for row, col in zip(rows.tolist(), cols.tolist()):
        union(row, col)

    groups: dict[int, list[int]] = {}
    for index in range(count):
        groups.setdefault(find(index), []).append(index)
    return list(groups.values())


def read_labels(path: Path) -> np.ndarray:
    if not path.exists() or path.stat().st_size == 0:
        return np.zeros((0, 5), dtype=np.float64)
    labels = np.loadtxt(path, dtype=np.float64, ndmin=2)
    if labels.shape[1] != 5:
        raise ValueError(f"{path}: expected 5 columns, got shape {labels.shape}")
    if not np.isfinite(labels).all():
        raise ValueError(f"{path}: labels contain NaN or infinity")
    if (labels[:, 0] < 0).any() or (labels[:, 0] % 1 != 0).any():
        raise ValueError(f"{path}: class IDs must be non-negative integers")
    if (labels[:, 1:] < 0).any() or (labels[:, 1:] > 1).any():
        raise ValueError(f"{path}: normalized box coordinates must be in [0, 1]")
    return labels


def merge_labels(labels: np.ndarray, threshold: float) -> tuple[list[tuple[list[int], np.ndarray]], int]:
    output: list[tuple[list[int], np.ndarray]] = []
    merged_groups = 0
    for indices in connected_groups(labels[:, 1:5], threshold):
        group = labels[indices]
        geometry = group[:, 1:5].mean(axis=0)
        classes = sorted({int(value) for value in group[:, 0]})
        output.append((classes, geometry))
        merged_groups += len(indices) > 1

    output.sort(key=lambda item: (*item[1].tolist(), item[0]))
    return output, merged_groups


def write_labels(path: Path, labels: list[tuple[list[int], np.ndarray]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not len(labels):
        path.write_text("", encoding="utf-8")
        return
    coordinate_format = f"{{:.{DECIMALS}f}}"
    lines = []
    for classes, geometry in labels:
        class_text = " ".join(str(class_id) for class_id in classes)
        geometry_text = " ".join(coordinate_format.format(value) for value in geometry)
        lines.append(f"{class_text} {geometry_text}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def transfer_image(source: Path, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if IMAGE_MODE == "copy":
        shutil.copy2(source, destination)
    elif IMAGE_MODE == "symlink":
        destination.symlink_to(source.resolve())
    elif IMAGE_MODE == "hardlink":
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
    else:
        raise ValueError(f"unknown IMAGE_MODE: {IMAGE_MODE!r}")


def prepare_output():
    if OUTPUT_ROOT.exists():
        if not OVERWRITE:
            raise FileExistsError(f"{OUTPUT_ROOT} already exists; set OVERWRITE=True to replace it")
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True)


def write_data_yaml(source_config: dict):
    config = {key: value for key, value in source_config.items() if key != "yaml_file"}
    config["path"] = str(OUTPUT_ROOT.resolve())
    config["train"] = "images/train"
    config["val"] = "images/val"
    if "test" in config:
        config["test"] = "images/test"
    (OUTPUT_ROOT / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def main():
    source_yaml = SOURCE_ROOT / "data.yaml"
    source_config = yaml.safe_load(source_yaml.read_text(encoding="utf-8")) or {}
    prepare_output()

    stats = Counter()
    splits = [split for split in ("train", "val", "test") if (SOURCE_ROOT / "images" / split).is_dir()]
    for split in splits:
        image_dir = SOURCE_ROOT / "images" / split
        label_dir = SOURCE_ROOT / "labels" / split
        images = sorted(path for path in image_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
        for image_path in images:
            relative = image_path.relative_to(image_dir)
            transfer_image(image_path, OUTPUT_ROOT / "images" / split / relative)

            label_path = (label_dir / relative).with_suffix(".txt")
            labels = read_labels(label_path)
            merged, merged_groups = merge_labels(labels, IOU_THRESHOLD)
            output_label = (OUTPUT_ROOT / "labels" / split / relative).with_suffix(".txt")
            write_labels(output_label, merged)

            stats["images"] += 1
            stats["input_rows"] += len(labels)
            stats["output_rows"] += len(merged)
            stats["class_assignments"] += sum(len(classes) for classes, _ in merged)
            stats["merged_groups"] += merged_groups
            stats["images_with_merges"] += merged_groups > 0

    write_data_yaml(source_config)
    print(f"Created: {OUTPUT_ROOT}")
    print(f"IoU threshold: > {IOU_THRESHOLD}")
    print(f"Images: {stats['images']}")
    print(f"Images with merged boxes: {stats['images_with_merges']}")
    print(f"Merged groups: {stats['merged_groups']}")
    print(f"Label rows: {stats['input_rows']} -> {stats['output_rows']}")
    print(f"Class assignments retained: {stats['class_assignments']}")


if __name__ == "__main__":
    main()
