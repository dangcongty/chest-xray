"""Audit VinDr-CXR bounding-box size habits for every radiologist.

Outputs per-radiologist and per-radiologist/class CSV files, a JSON report,
plots, and a montage of suspiciously large focal-lesion annotations.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom


ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS = ROOT / "datasets/dicom/train_with_size.csv"
DICOM_DIR = ROOT / "datasets/dicom/train"
OUTPUT_DIR = ROOT / "runs/eda/radiologists"

IQR_FACTOR = 1.5
TOP_AREA_QUANTILE = 0.95
ABSOLUTE_LARGE_AREA = 0.25
LUNG_WIDTH = 0.35
LUNG_HEIGHT = 0.50
FOCAL_CLASSES = {"Nodule/Mass", "Calcification"}
FOCAL_LARGE_AREA = 0.10
MONTAGE_COUNT = 20
MONTAGE_TILE = 420


def load_annotations() -> tuple[pd.DataFrame, pd.DataFrame]:
    all_rows = pd.read_csv(ANNOTATIONS)
    boxes = all_rows.dropna(subset=["x_min", "y_min", "x_max", "y_max"]).copy()
    boxes["box_width"] = (boxes["x_max"] - boxes["x_min"]) / boxes["width"]
    boxes["box_height"] = (boxes["y_max"] - boxes["y_min"]) / boxes["height"]
    boxes["box_area"] = boxes["box_width"] * boxes["box_height"]
    boxes["aspect_ratio"] = boxes["box_width"] / boxes["box_height"].clip(lower=1e-12)
    if not boxes[["box_width", "box_height", "box_area"]].apply(np.isfinite).all().all():
        raise ValueError("non-finite normalized box dimensions")
    return all_rows, boxes


def mark_large_boxes(boxes: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    boxes = boxes.copy()
    fences = {}
    boxes["class_size_outlier"] = False
    boxes["class_top_area"] = False
    for class_name, group in boxes.groupby("class_name"):
        q1 = group[["box_width", "box_height"]].quantile(0.25)
        q3 = group[["box_width", "box_height"]].quantile(0.75)
        upper = (q3 + IQR_FACTOR * (q3 - q1)).clip(upper=1.0)
        area_threshold = float(group["box_area"].quantile(TOP_AREA_QUANTILE))
        boxes.loc[group.index, "class_size_outlier"] = (
            (group["box_width"] > upper["box_width"])
            | (group["box_height"] > upper["box_height"])
        )
        boxes.loc[group.index, "class_top_area"] = group["box_area"] >= area_threshold
        fences[class_name] = {
            "count": len(group),
            "upper_width": float(upper["box_width"]),
            "upper_height": float(upper["box_height"]),
            "top_5pct_area_threshold": area_threshold,
        }

    boxes["absolute_large"] = boxes["box_area"] >= ABSOLUTE_LARGE_AREA
    boxes["lung_sized"] = (
        ((boxes["box_width"] >= LUNG_WIDTH) & (boxes["box_height"] >= LUNG_HEIGHT))
        | boxes["absolute_large"]
    )
    boxes["suspicious_focal_large"] = boxes["class_name"].isin(FOCAL_CLASSES) & (
        (boxes["box_area"] >= FOCAL_LARGE_AREA)
        | (boxes["box_width"] >= LUNG_WIDTH)
        | (boxes["box_height"] >= LUNG_HEIGHT)
    )
    return boxes, fences


def summarize_radiologists(all_rows: pd.DataFrame, boxes: pd.DataFrame) -> pd.DataFrame:
    total_rows = all_rows.groupby("rad_id").size().rename("all_rows")
    no_finding = (all_rows["class_name"] == "No finding").groupby(all_rows["rad_id"]).sum().rename("no_finding")
    summary = boxes.groupby("rad_id").agg(
        positive_boxes=("box_area", "size"),
        positive_images=("image_id", "nunique"),
        median_area=("box_area", "median"),
        p95_area=("box_area", lambda values: values.quantile(0.95)),
        median_width=("box_width", "median"),
        median_height=("box_height", "median"),
        class_size_outliers=("class_size_outlier", "sum"),
        top_5pct_area=("class_top_area", "sum"),
        absolute_large=("absolute_large", "sum"),
        lung_sized=("lung_sized", "sum"),
        suspicious_focal_large=("suspicious_focal_large", "sum"),
    ).reindex(total_rows.index)
    summary = summary.join(total_rows).join(no_finding).fillna(0)

    for column in (
        "class_size_outliers", "top_5pct_area", "absolute_large", "lung_sized", "suspicious_focal_large"
    ):
        summary[f"{column}_rate"] = summary[column] / summary["positive_boxes"].replace(0, np.nan)

    # Expected outlier count accounts for each radiologist's class mix.
    class_rates = boxes.groupby("class_name")["class_size_outlier"].mean()
    expected = boxes.assign(expected=boxes["class_name"].map(class_rates)).groupby("rad_id")["expected"].sum()
    summary["expected_class_size_outliers"] = expected
    summary["outlier_observed_expected_ratio"] = (
        summary["class_size_outliers"] / summary["expected_class_size_outliers"].replace(0, np.nan)
    )
    return summary.reset_index().sort_values(
        ["outlier_observed_expected_ratio", "positive_boxes"], ascending=[False, False]
    )


def summarize_by_class(boxes: pd.DataFrame) -> pd.DataFrame:
    output = boxes.groupby(["rad_id", "class_name"]).agg(
        boxes=("box_area", "size"),
        images=("image_id", "nunique"),
        median_width=("box_width", "median"),
        median_height=("box_height", "median"),
        median_area=("box_area", "median"),
        p95_area=("box_area", lambda values: values.quantile(0.95)),
        class_size_outliers=("class_size_outlier", "sum"),
        lung_sized=("lung_sized", "sum"),
        suspicious_focal_large=("suspicious_focal_large", "sum"),
    ).reset_index()
    output["class_size_outlier_rate"] = output["class_size_outliers"] / output["boxes"]
    return output.sort_values(["rad_id", "boxes"], ascending=[True, False])


def save_plots(summary: pd.DataFrame, boxes: pd.DataFrame):
    positive = summary[summary["positive_boxes"] > 0].copy()
    positive = positive.sort_values("outlier_observed_expected_ratio", ascending=False)
    fig, axes = plt.subplots(2, 1, figsize=(13, 10), constrained_layout=True)
    axes[0].bar(positive["rad_id"], positive["class_size_outliers_rate"] * 100)
    axes[0].set_ylabel("Class-adjusted size outliers (%)")
    axes[0].set_title("Large-box rate by radiologist (class-specific Tukey fences)")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(positive["rad_id"], positive["outlier_observed_expected_ratio"])
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Observed / expected outliers")
    axes[1].set_title("Adjusted for each radiologist's class mix")
    axes[1].grid(axis="y", alpha=0.25)
    fig.savefig(OUTPUT_DIR / "large_box_rates_by_radiologist.png", dpi=180)
    plt.close(fig)

    ordered = positive["rad_id"].tolist()
    data = [np.log10(boxes.loc[boxes["rad_id"] == rad, "box_area"].clip(lower=1e-7)) for rad in ordered]
    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
    ax.boxplot(data, tick_labels=ordered, showfliers=False)
    ax.set_ylabel("log10(normalized box area)")
    ax.set_title("Box-area distribution by radiologist")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(OUTPUT_DIR / "box_area_distribution_by_radiologist.png", dpi=180)
    plt.close(fig)


def read_dicom(path: Path) -> np.ndarray:
    dataset = pydicom.dcmread(path)
    pixels = dataset.pixel_array.astype(np.float32)
    low, high = np.percentile(pixels, (0.5, 99.5))
    pixels = np.clip((pixels - low) / max(high - low, 1e-6), 0, 1)
    if getattr(dataset, "PhotometricInterpretation", "") == "MONOCHROME1":
        pixels = 1.0 - pixels
    return (pixels * 255).astype(np.uint8)


def make_montage(boxes: pd.DataFrame):
    suspicious = boxes[boxes["suspicious_focal_large"]].sort_values("box_area", ascending=False).head(MONTAGE_COUNT)
    if suspicious.empty:
        return []
    tiles, records = [], []
    for row in suspicious.itertuples():
        dicom_path = DICOM_DIR / f"{row.image_id}.dicom"
        if not dicom_path.exists():
            continue
        gray = read_dicom(dicom_path)
        tile = cv2.cvtColor(cv2.resize(gray, (MONTAGE_TILE, MONTAGE_TILE)), cv2.COLOR_GRAY2BGR)
        sx, sy = MONTAGE_TILE / row.width, MONTAGE_TILE / row.height
        p1 = (round(row.x_min * sx), round(row.y_min * sy))
        p2 = (round(row.x_max * sx), round(row.y_max * sy))
        cv2.rectangle(tile, p1, p2, (0, 0, 255), 3)
        cv2.rectangle(tile, (0, 0), (MONTAGE_TILE, 54), (0, 0, 0), -1)
        cv2.putText(tile, f"{row.rad_id} | {row.class_name}", (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(tile, f"area={row.box_area:.3f}  w={row.box_width:.2f} h={row.box_height:.2f}",
                    (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
        records.append({
            "image_id": row.image_id, "rad_id": row.rad_id, "class_name": row.class_name,
            "box_width": row.box_width, "box_height": row.box_height, "box_area": row.box_area,
        })
    if tiles:
        columns = 4
        while len(tiles) % columns:
            tiles.append(np.zeros_like(tiles[0]))
        rows = [np.concatenate(tiles[index:index + columns], axis=1) for index in range(0, len(tiles), columns)]
        cv2.imwrite(str(OUTPUT_DIR / "suspicious_focal_large_montage.jpg"), np.concatenate(rows, axis=0))
    return records


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows, boxes = load_annotations()
    boxes, fences = mark_large_boxes(boxes)
    summary = summarize_radiologists(all_rows, boxes)
    by_class = summarize_by_class(boxes)
    summary.to_csv(OUTPUT_DIR / "radiologist_summary.csv", index=False)
    by_class.to_csv(OUTPUT_DIR / "radiologist_class_summary.csv", index=False)
    boxes[boxes["suspicious_focal_large"]].sort_values("box_area", ascending=False).to_csv(
        OUTPUT_DIR / "suspicious_focal_annotations.csv", index=False
    )
    save_plots(summary, boxes)
    montage_records = make_montage(boxes)

    report = {
        "definitions": {
            "class_size_outlier": f"width or height > Q3 + {IQR_FACTOR}*IQR within the same class",
            "absolute_large": f"normalized box area >= {ABSOLUTE_LARGE_AREA}",
            "lung_sized": f"(width >= {LUNG_WIDTH} and height >= {LUNG_HEIGHT}) or absolute_large",
            "suspicious_focal_large": (
                f"class in {sorted(FOCAL_CLASSES)} and "
                f"(area >= {FOCAL_LARGE_AREA} or width >= {LUNG_WIDTH} or height >= {LUNG_HEIGHT})"
            ),
        },
        "rows": len(all_rows),
        "positive_boxes": len(boxes),
        "radiologists": len(summary),
        "radiologists_with_positive_boxes": int((summary["positive_boxes"] > 0).sum()),
        "suspicious_focal_large_count": int(boxes["suspicious_focal_large"].sum()),
        "class_fences": fences,
        "radiologist_summary": summary.replace({np.nan: None}).to_dict(orient="records"),
        "montage_annotations": montage_records,
    }
    (OUTPUT_DIR / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"\nSuspicious focal large boxes: {report['suspicious_focal_large_count']}")
    print(f"Saved: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
