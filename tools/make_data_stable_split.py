"""Rebuild the YOLO dataset with all per-class box-size outliers in train.

An image is forced into train when at least one box has an outlier width OR
height for that class. A fixed-size validation subset is then selected from the
remaining images by multilabel stratification so every class approaches the
requested train/validation label ratio. Splitting is image-level, preventing
the same radiograph from leaking into both splits.

Outliers are detected independently per class and dimension using Tukey fences:
    lower = Q1 - IQR_FACTOR * IQR
    upper = Q3 + IQR_FACTOR * IQR

Run:
    ./lib/bin/python tools/make_data_stable_split.py
"""

from __future__ import annotations

import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml


# -----------------------------------------------------------------------------
# Global settings
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "datasets/yolo"
OUTPUT_ROOT = ROOT / "datasets/yolo_stable_split"
SOURCE_SPLITS = ("train", "val")
IQR_FACTOR = 1.5
TRAIN_RATIO = 0.80
SEED = 1234
SWAP_ITERATIONS = 250_000
IMAGE_MODE = "hardlink"  # hardlink | copy | symlink
OVERWRITE = True

IMAGE_SUFFIXES = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}


def read_labels(path: Path, nc: int) -> np.ndarray:
    if not path.exists() or path.stat().st_size == 0:
        return np.zeros((0, 5), dtype=np.float64)
    labels = np.loadtxt(path, dtype=np.float64, ndmin=2)
    if labels.shape[1] != 5:
        raise ValueError(f"{path}: expected 5 columns, got {labels.shape[1]}")
    if not np.isfinite(labels).all():
        raise ValueError(f"{path}: contains NaN or infinity")
    if (labels[:, 0] < 0).any() or (labels[:, 0] >= nc).any() or (labels[:, 0] % 1 != 0).any():
        raise ValueError(f"{path}: invalid class ID")
    if (labels[:, 1:] < 0).any() or (labels[:, 1:] > 1).any():
        raise ValueError(f"{path}: normalized box values must be in [0, 1]")
    return labels


def collect_records(nc: int) -> list[dict]:
    records = []
    seen_names: dict[str, Path] = {}
    for source_split in SOURCE_SPLITS:
        image_root = SOURCE_ROOT / "images" / source_split
        label_root = SOURCE_ROOT / "labels" / source_split
        for image in sorted(path for path in image_root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES):
            relative = image.relative_to(image_root)
            key = str(relative.with_suffix(""))
            if key in seen_names:
                raise ValueError(f"duplicate image identity across source splits: {seen_names[key]} and {image}")
            seen_names[key] = image
            label = (label_root / relative).with_suffix(".txt")
            records.append(
                {
                    "source_split": source_split,
                    "image": image,
                    "label": label,
                    "relative": relative,
                    "labels": read_labels(label, nc),
                }
            )
    if not records:
        raise FileNotFoundError(f"no images found below {SOURCE_ROOT}")
    return records


def calculate_fences(records: list[dict], nc: int) -> dict[int, dict[str, np.ndarray | int]]:
    dimensions: dict[int, list[np.ndarray]] = defaultdict(list)
    for record in records:
        for row in record["labels"]:
            dimensions[int(row[0])].append(row[3:5])  # normalized width, height

    fences = {}
    for class_id in range(nc):
        values = np.asarray(dimensions[class_id], dtype=np.float64)
        if not len(values):
            raise ValueError(f"class {class_id} has no labels; cannot calculate an outlier fence")
        q1, q3 = np.quantile(values, (0.25, 0.75), axis=0)
        iqr = q3 - q1
        fences[class_id] = {
            "count": len(values),
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower": np.maximum(0.0, q1 - IQR_FACTOR * iqr),
            "upper": np.minimum(1.0, q3 + IQR_FACTOR * iqr),
        }
    return fences


def mark_outliers(records: list[dict], fences: dict[int, dict]):
    for record in records:
        flags = []
        reasons = []
        for row in record["labels"]:
            class_id = int(row[0])
            width_height = row[3:5]
            fence = fences[class_id]
            low = width_height < fence["lower"]
            high = width_height > fence["upper"]
            flags.append(bool(np.any(low | high)))
            reasons.append(
                {
                    "class_id": class_id,
                    "width_low": bool(low[0]),
                    "width_high": bool(high[0]),
                    "height_low": bool(low[1]),
                    "height_high": bool(high[1]),
                }
            )
        record["outlier_flags"] = np.asarray(flags, dtype=bool)
        record["outlier_reasons"] = reasons
        record["has_outlier"] = bool(any(flags))


def label_counts(record: dict, nc: int) -> np.ndarray:
    if not len(record["labels"]):
        return np.zeros(nc, dtype=np.int64)
    return np.bincount(record["labels"][:, 0].astype(np.int64), minlength=nc)


def stratified_split(records: list[dict], nc: int) -> dict:
    """Select a fixed-size, outlier-free validation set with balanced label counts."""
    total_images = len(records)
    train_size = round(total_images * TRAIN_RATIO)
    val_size = total_images - train_size
    mandatory_train = [index for index, record in enumerate(records) if record["has_outlier"]]
    eligible = [index for index, record in enumerate(records) if not record["has_outlier"]]
    if len(mandatory_train) > train_size:
        raise ValueError(
            f"{len(mandatory_train)} outlier images exceed requested train size {train_size}; "
            "increase TRAIN_RATIO or reduce IQR_FACTOR"
        )
    if len(eligible) < val_size:
        raise ValueError(f"only {len(eligible)} outlier-free images are available for val_size={val_size}")

    all_counts = np.stack([label_counts(record, nc) for record in records])
    eligible_counts = all_counts[eligible]
    total_class_counts = all_counts.sum(0)
    target_val_counts = total_class_counts * (1.0 - TRAIN_RATIO)
    denominator = np.maximum(target_val_counts, 1.0)
    rng = np.random.default_rng(SEED)

    # Greedily fill deficits first. Small random jitter only resolves exact ties
    # and remains reproducible through SEED.
    selected_mask = np.zeros(len(eligible), dtype=bool)
    current = np.zeros(nc, dtype=np.int64)
    for _ in range(val_size):
        candidates = np.flatnonzero(~selected_mask)
        proposed = current[None, :] + eligible_counts[candidates]
        scores = (((proposed - target_val_counts) / denominator) ** 2).mean(1)
        scores += rng.random(len(scores)) * 1e-12
        chosen = candidates[int(scores.argmin())]
        selected_mask[chosen] = True
        current += eligible_counts[chosen]

    # Improve the class-count match while preserving validation size and the
    # hard constraint that no outlier image may enter validation.
    selected = np.flatnonzero(selected_mask)
    unselected = np.flatnonzero(~selected_mask)

    def objective(counts: np.ndarray) -> float:
        return float((((counts - target_val_counts) / denominator) ** 2).mean())

    score = objective(current)
    accepted_swaps = 0
    for _ in range(SWAP_ITERATIONS):
        selected_position = int(rng.integers(len(selected)))
        unselected_position = int(rng.integers(len(unselected)))
        remove_index = selected[selected_position]
        add_index = unselected[unselected_position]
        proposed = current - eligible_counts[remove_index] + eligible_counts[add_index]
        proposed_score = objective(proposed)
        if proposed_score + 1e-15 < score:
            current, score = proposed, proposed_score
            selected[selected_position], unselected[unselected_position] = add_index, remove_index
            accepted_swaps += 1

    validation_record_indices = {eligible[index] for index in selected.tolist()}
    for index, record in enumerate(records):
        record["destination_split"] = "val" if index in validation_record_indices else "train"

    return {
        "requested_train_ratio": TRAIN_RATIO,
        "target_train_images": train_size,
        "target_val_images": val_size,
        "mandatory_outlier_train_images": len(mandatory_train),
        "stratification_objective": score,
        "accepted_swaps": accepted_swaps,
        "target_val_class_counts": target_val_counts,
        "actual_val_class_counts": current,
    }


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
        raise ValueError(f"unsupported IMAGE_MODE={IMAGE_MODE!r}")


def prepare_output():
    if OUTPUT_ROOT.exists():
        if not OVERWRITE:
            raise FileExistsError(f"{OUTPUT_ROOT} exists; set OVERWRITE=True to replace it")
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True)


def materialize(records: list[dict]):
    prepare_output()
    for record in records:
        split = record["destination_split"]
        relative = record["relative"]
        image_destination = OUTPUT_ROOT / "images" / split / relative
        label_destination = (OUTPUT_ROOT / "labels" / split / relative).with_suffix(".txt")
        transfer_image(record["image"], image_destination)
        label_destination.parent.mkdir(parents=True, exist_ok=True)
        if record["label"].exists():
            shutil.copy2(record["label"], label_destination)
        else:
            label_destination.write_text("", encoding="utf-8")


def serializable_fence(fence: dict) -> dict:
    return {
        "count": int(fence["count"]),
        "q1_width": float(fence["q1"][0]),
        "q1_height": float(fence["q1"][1]),
        "q3_width": float(fence["q3"][0]),
        "q3_height": float(fence["q3"][1]),
        "lower_width": float(fence["lower"][0]),
        "lower_height": float(fence["lower"][1]),
        "upper_width": float(fence["upper"][0]),
        "upper_height": float(fence["upper"][1]),
    }


def build_report(records: list[dict], fences: dict, names: list[str], split_info: dict) -> dict:
    image_counts = Counter(record["destination_split"] for record in records)
    box_counts = Counter()
    outlier_counts = Counter()
    movement = Counter()
    class_stats = {
        class_id: {"name": names[class_id], "train": 0, "val": 0, "outliers": 0}
        for class_id in range(len(names))
    }
    reason_counts = Counter()

    for record in records:
        split = record["destination_split"]
        movement[f"{record['source_split']}->{split}"] += 1
        box_counts[split] += len(record["labels"])
        for row, is_outlier, reason in zip(
            record["labels"], record["outlier_flags"], record["outlier_reasons"]
        ):
            class_id = int(row[0])
            class_stats[class_id][split] += 1
            class_stats[class_id]["outliers"] += int(is_outlier)
            outlier_counts[split] += int(is_outlier)
            for key in ("width_low", "width_high", "height_low", "height_high"):
                reason_counts[key] += int(reason[key])

    total_images = len(records)
    total_boxes = sum(box_counts.values())
    target_val_counts = split_info["target_val_class_counts"]
    for class_id, values in class_stats.items():
        total = values["train"] + values["val"]
        values["total"] = total
        values["train_ratio"] = values["train"] / total if total else 0.0
        values["val_ratio"] = values["val"] / total if total else 0.0
        values["outlier_ratio"] = values["outliers"] / total if total else 0.0
        values["target_val"] = float(target_val_counts[class_id])
        values["val_count_error"] = float(values["val"] - target_val_counts[class_id])
        values["val_ratio_error"] = values["val_ratio"] - (1.0 - TRAIN_RATIO)

    return {
        "source": str(SOURCE_ROOT),
        "output": str(OUTPUT_ROOT),
        "method": (
            "per-class width/height Tukey IQR fences; outlier images forced to train; "
            "fixed-ratio multilabel stratification on remaining images"
        ),
        "iqr_factor": IQR_FACTOR,
        "requested_train_ratio": TRAIN_RATIO,
        "seed": SEED,
        "stratification": {
            "target_train_images": split_info["target_train_images"],
            "target_val_images": split_info["target_val_images"],
            "mandatory_outlier_train_images": split_info["mandatory_outlier_train_images"],
            "objective": split_info["stratification_objective"],
            "accepted_swaps": split_info["accepted_swaps"],
        },
        "images": {
            "total": total_images,
            "train": image_counts["train"],
            "val": image_counts["val"],
            "train_ratio": image_counts["train"] / total_images,
            "val_ratio": image_counts["val"] / total_images,
        },
        "boxes": {
            "total": total_boxes,
            "train": box_counts["train"],
            "val": box_counts["val"],
            "train_ratio": box_counts["train"] / total_boxes,
            "val_ratio": box_counts["val"] / total_boxes,
            "outlier_total": sum(outlier_counts.values()),
            "outliers_in_train": outlier_counts["train"],
            "outliers_in_val": outlier_counts["val"],
        },
        "outlier_reasons": dict(reason_counts),
        "source_to_destination": dict(movement),
        "classes": {str(key): value for key, value in class_stats.items()},
        "fences": {str(key): serializable_fence(value) for key, value in fences.items()},
    }


def write_data_yaml(source_config: dict):
    config = {key: value for key, value in source_config.items() if key not in {"path", "train", "val", "test"}}
    config.update(path=str(OUTPUT_ROOT.resolve()), train="images/train", val="images/val")
    (OUTPUT_ROOT / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def main():
    source_config = yaml.safe_load((SOURCE_ROOT / "data.yaml").read_text(encoding="utf-8")) or {}
    raw_names = source_config.get("names", {})
    names = [raw_names[key] for key in sorted(raw_names, key=lambda value: int(value))] if isinstance(raw_names, dict) else raw_names
    if not names:
        raise ValueError("source data.yaml must contain class names")

    records = collect_records(len(names))
    fences = calculate_fences(records, len(names))
    mark_outliers(records, fences)
    split_info = stratified_split(records, len(names))
    materialize(records)
    write_data_yaml(source_config)
    report = build_report(records, fences, names, split_info)
    report_path = OUTPUT_ROOT / "split_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Created: {OUTPUT_ROOT}")
    print(f"Images: train={report['images']['train']} ({report['images']['train_ratio']:.2%}), "
          f"val={report['images']['val']} ({report['images']['val_ratio']:.2%})")
    print(f"Boxes: train={report['boxes']['train']} ({report['boxes']['train_ratio']:.2%}), "
          f"val={report['boxes']['val']} ({report['boxes']['val_ratio']:.2%})")
    print(f"Outlier boxes: {report['boxes']['outlier_total']} "
          f"(train={report['boxes']['outliers_in_train']}, val={report['boxes']['outliers_in_val']})")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
