"""EDA for highly overlapping annotations in a YOLO-format dataset.

For every IoU threshold, report:
  - matched_annotations: label rows having at least one overlapping box;
  - matched_pairs: unique pairs of overlapping boxes;
  - matched_images: images containing at least one overlapping pair;
  - same/different-class pair counts.

Run:
    ./lib/bin/python tools/eda.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "datasets/yolo_multi_class"
LABEL_ROOT = DATASET_ROOT / "labels"
IOU_THRESHOLDS = (0.80, 0.90, 0.95)
STRICTLY_GREATER = True
OUTPUT_JSON = DATASET_ROOT / "overlap_eda.json"


def read_labels(path: Path) -> tuple[list[set[int]], np.ndarray]:
    if path.stat().st_size == 0:
        return [], np.zeros((0, 4), dtype=np.float64)
    classes, boxes = [], []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        values = [float(value) for value in line.replace(",", " ").split()]
        if len(values) < 5:
            raise ValueError(f"{path}:{line_number}: expected classes followed by x y w h")
        classes.append({int(value) for value in values[:-4]})
        boxes.append(values[-4:])
    return classes, np.asarray(boxes, dtype=np.float64)


def pairwise_iou(xywh: np.ndarray) -> np.ndarray:
    xyxy = np.empty_like(xywh)
    xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
    xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
    xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
    xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2

    top_left = np.maximum(xyxy[:, None, :2], xyxy[None, :, :2])
    bottom_right = np.minimum(xyxy[:, None, 2:], xyxy[None, :, 2:])
    wh = np.clip(bottom_right - top_left, 0, None)
    intersection = wh[..., 0] * wh[..., 1]
    area = np.clip(xyxy[:, 2] - xyxy[:, 0], 0, None) * np.clip(xyxy[:, 3] - xyxy[:, 1], 0, None)
    union = area[:, None] + area[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def empty_stats() -> dict:
    return {
        "files": 0,
        "annotations": 0,
        "empty_files": 0,
        "thresholds": {
            str(threshold): {
                "matched_annotations": 0,
                "matched_pairs": 0,
                "matched_images": 0,
                "same_class_pairs": 0,
                "different_class_pairs": 0,
            }
            for threshold in IOU_THRESHOLDS
        },
    }


def analyze_files(files: list[Path]) -> dict:
    stats = empty_stats()
    stats["files"] = len(files)
    for path in files:
        classes, boxes = read_labels(path)
        count = len(boxes)
        stats["annotations"] += count
        stats["empty_files"] += count == 0
        if count < 2:
            continue

        ious = pairwise_iou(boxes)
        upper_triangle = np.triu(np.ones((count, count), dtype=bool), k=1)
        same_class = np.asarray(
            [[bool(classes[row] & classes[col]) for col in range(count)] for row in range(count)], dtype=bool
        )
        for threshold in IOU_THRESHOLDS:
            overlap = ious > threshold if STRICTLY_GREATER else ious >= threshold
            pairs = overlap & upper_triangle
            pair_rows, pair_cols = np.where(pairs)
            if not len(pair_rows):
                continue
            matched = np.zeros(count, dtype=bool)
            matched[pair_rows] = True
            matched[pair_cols] = True
            result = stats["thresholds"][str(threshold)]
            result["matched_annotations"] += int(matched.sum())
            result["matched_pairs"] += int(pairs.sum())
            result["matched_images"] += 1
            result["same_class_pairs"] += int((pairs & same_class).sum())
            result["different_class_pairs"] += int((pairs & ~same_class).sum())
    return stats


def combine(parts: dict[str, dict]) -> dict:
    total = empty_stats()
    for stats in parts.values():
        for key in ("files", "annotations", "empty_files"):
            total[key] += stats[key]
        for threshold in total["thresholds"]:
            for key in total["thresholds"][threshold]:
                total["thresholds"][threshold][key] += stats["thresholds"][threshold][key]
    return total


def print_stats(name: str, stats: dict):
    print(f"\n{name}: files={stats['files']:,}, annotations={stats['annotations']:,}, empty={stats['empty_files']:,}")
    print("IoU       annotations       pairs      images    same-cls    diff-cls")
    for threshold, values in stats["thresholds"].items():
        percentage = 100 * values["matched_annotations"] / max(stats["annotations"], 1)
        print(
            f"> {float(threshold):.2f}  "
            f"{values['matched_annotations']:>9,} ({percentage:6.2f}%)  "
            f"{values['matched_pairs']:>9,}  "
            f"{values['matched_images']:>9,}  "
            f"{values['same_class_pairs']:>9,}  "
            f"{values['different_class_pairs']:>9,}"
        )


def main():
    if not LABEL_ROOT.is_dir():
        raise FileNotFoundError(f"label directory not found: {LABEL_ROOT}")
    split_dirs = sorted(path for path in LABEL_ROOT.iterdir() if path.is_dir())
    results = {
        split_dir.name: analyze_files(sorted(split_dir.rglob("*.txt")))
        for split_dir in split_dirs
    }
    results["total"] = combine(results)
    for name, stats in results.items():
        print_stats(name, stats)
    OUTPUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
