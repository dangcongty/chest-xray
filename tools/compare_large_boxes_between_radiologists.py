"""Compare suspiciously large focal boxes with the other annotators on each image."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom


ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS = ROOT / "datasets/dicom/train_with_size.csv"
DICOM_DIR = ROOT / "datasets/dicom/train"
OUTPUT_DIR = ROOT / "runs/eda/radiologists"
FOCAL_CLASSES = {"Nodule/Mass", "Calcification"}
LARGE_AREA = 0.10
LARGE_WIDTH = 0.35
LARGE_HEIGHT = 0.50
SAME_LOCATION_IOBB = 0.50
TILE_SIZE = 520
MONTAGE_COUNT = 16


def add_geometry(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["x_min", "y_min", "x_max", "y_max"]).copy()
    df["box_width"] = (df.x_max - df.x_min) / df.width
    df["box_height"] = (df.y_max - df.y_min) / df.height
    df["box_area"] = df.box_width * df.box_height
    return df


def iobb(anchor, boxes: pd.DataFrame) -> np.ndarray:
    """Intersection divided by each candidate box area (containment score)."""
    if boxes.empty:
        return np.empty(0, dtype=np.float64)
    ix1 = np.maximum(float(anchor.x_min), boxes.x_min.to_numpy())
    iy1 = np.maximum(float(anchor.y_min), boxes.y_min.to_numpy())
    ix2 = np.minimum(float(anchor.x_max), boxes.x_max.to_numpy())
    iy2 = np.minimum(float(anchor.y_max), boxes.y_max.to_numpy())
    intersection = np.maximum(ix2 - ix1, 0) * np.maximum(iy2 - iy1, 0)
    area = (boxes.x_max - boxes.x_min) * (boxes.y_max - boxes.y_min)
    return intersection / np.maximum(area.to_numpy(), 1e-12)


def compare(boxes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    large = boxes[
        boxes.class_name.isin(FOCAL_CLASSES)
        & ((boxes.box_area >= LARGE_AREA) | (boxes.box_width >= LARGE_WIDTH) | (boxes.box_height >= LARGE_HEIGHT))
    ].sort_values("box_area", ascending=False)
    detail = []
    for anchor in large.itertuples():
        image_rows = boxes[boxes.image_id == anchor.image_id]
        for other_rad in sorted(set(image_rows.rad_id) - {anchor.rad_id}):
            other = image_rows[image_rows.rad_id == other_rad].copy()
            other["iobb_anchor"] = iobb(anchor, other)
            local = other[other.iobb_anchor >= SAME_LOCATION_IOBB]
            same = local[local.class_name == anchor.class_name]
            area_ratios = same.box_area / anchor.box_area
            detail.append({
                "image_id": anchor.image_id,
                "anchor_rad": anchor.rad_id,
                "other_rad": other_rad,
                "class_name": anchor.class_name,
                "anchor_area": anchor.box_area,
                "anchor_x_min": anchor.x_min,
                "anchor_y_min": anchor.y_min,
                "anchor_x_max": anchor.x_max,
                "anchor_y_max": anchor.y_max,
                "same_class_boxes_at_location": len(same),
                "same_class_max_area_ratio": area_ratios.max() if len(area_ratios) else np.nan,
                "same_class_median_area_ratio": area_ratios.median() if len(area_ratios) else np.nan,
                "same_class_max_iobb": same.iobb_anchor.max() if len(same) else np.nan,
                "other_classes_at_location": "|".join(sorted(set(local.class_name) - {anchor.class_name})),
            })
    detail = pd.DataFrame(detail)
    summary = detail.groupby(["anchor_rad", "class_name"], dropna=False).agg(
        comparisons=("image_id", "size"),
        same_class_present=("same_class_boxes_at_location", lambda x: int((x > 0).sum())),
        multiple_same_class_boxes=("same_class_boxes_at_location", lambda x: int((x > 1).sum())),
        median_same_class_count=("same_class_boxes_at_location", "median"),
        median_max_area_ratio=("same_class_max_area_ratio", "median"),
        different_class_present=("other_classes_at_location", lambda x: int((x != "").sum())),
    ).reset_index()
    summary["same_class_present_rate"] = summary.same_class_present / summary.comparisons
    summary["multiple_boxes_rate"] = summary.multiple_same_class_boxes / summary.comparisons
    summary["different_class_present_rate"] = summary.different_class_present / summary.comparisons
    return detail, summary


def read_dicom(path: Path) -> np.ndarray:
    ds = pydicom.dcmread(path)
    pixels = ds.pixel_array.astype(np.float32)
    lo, hi = np.percentile(pixels, (0.5, 99.5))
    pixels = np.clip((pixels - lo) / max(hi - lo, 1e-6), 0, 1)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        pixels = 1 - pixels
    return (pixels * 255).astype(np.uint8)


def make_overlay(boxes: pd.DataFrame):
    large = boxes[
        boxes.class_name.isin(FOCAL_CLASSES)
        & ((boxes.box_area >= LARGE_AREA) | (boxes.box_width >= LARGE_WIDTH) | (boxes.box_height >= LARGE_HEIGHT))
    ].sort_values("box_area", ascending=False).head(MONTAGE_COUNT)
    colors = [(0, 0, 255), (0, 255, 0), (255, 255, 0)]
    tiles = []
    for anchor in large.itertuples():
        path = DICOM_DIR / f"{anchor.image_id}.dicom"
        if not path.exists():
            continue
        tile = cv2.cvtColor(cv2.resize(read_dicom(path), (TILE_SIZE, TILE_SIZE)), cv2.COLOR_GRAY2BGR)
        image_rows = boxes[boxes.image_id == anchor.image_id]
        rad_ids = [anchor.rad_id] + sorted(set(image_rows.rad_id) - {anchor.rad_id})
        for rad_index, rad_id in enumerate(rad_ids):
            rows = image_rows[image_rows.rad_id == rad_id].copy()
            if rad_id != anchor.rad_id:
                rows["containment"] = iobb(anchor, rows)
                rows = rows[(rows.containment >= SAME_LOCATION_IOBB) & (rows.class_name == anchor.class_name)]
            else:
                rows = rows[
                    (rows.class_name == anchor.class_name)
                    & (rows.x_min == anchor.x_min) & (rows.y_min == anchor.y_min)
                    & (rows.x_max == anchor.x_max) & (rows.y_max == anchor.y_max)
                ]
            color = colors[min(rad_index, len(colors) - 1)]
            for row in rows.itertuples():
                sx, sy = TILE_SIZE / row.width, TILE_SIZE / row.height
                cv2.rectangle(tile, (round(row.x_min*sx), round(row.y_min*sy)),
                              (round(row.x_max*sx), round(row.y_max*sy)), color, 2)
        cv2.rectangle(tile, (0, 0), (TILE_SIZE, 61), (0, 0, 0), -1)
        cv2.putText(tile, f"{anchor.class_name} | large: {anchor.rad_id} (red)", (8, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)
        legend = " | ".join(f"{r}: {'red' if i == 0 else 'green' if i == 1 else 'cyan'}" for i, r in enumerate(rad_ids))
        cv2.putText(tile, legend, (8, 49), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    if tiles:
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        canvas = np.concatenate([np.concatenate(tiles[i:i+4], axis=1) for i in range(0, len(tiles), 4)], axis=0)
        cv2.imwrite(str(OUTPUT_DIR / "large_box_other_radiologists_overlay.jpg"), canvas)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    boxes = add_geometry(pd.read_csv(ANNOTATIONS))
    detail, summary = compare(boxes)
    detail.to_csv(OUTPUT_DIR / "large_box_other_radiologists_detail.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "large_box_other_radiologists_summary.csv", index=False)
    make_overlay(boxes)
    print(summary.to_string(index=False))
    print(f"\nSaved comparison and overlay to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
