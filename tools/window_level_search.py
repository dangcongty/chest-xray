"""End-to-end DICOM window/level search followed by top-3 window stacking.

The split and labels are inherited verbatim from datasets/yolo. Search trials
train identical image-only YOLO26 models and are ranked by best validation
mAP50. After search, the three best sufficiently different windows are stacked
as RGB channels and used for a final training run.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pydicom
import yaml
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
BASE_DATA = ROOT / "datasets/yolo/data.yaml"
DICOM_DIR = ROOT / "datasets/dicom/train"
OUTPUT_ROOT = ROOT / "datasets/window_level_search"
RUNS_ROOT = ROOT / "runs/window_level_search"
WEIGHTS = ROOT / "yolo26m.pt"

NUM_TRIALS = 8
SEARCH_EPOCHS = 10
FINAL_EPOCHS = 500
IMAGE_SIZE = 640
BATCH_SIZE = 16
WORKERS = 8
DEVICE: str | int = 0
MODEL_SIZE = "m"
SEED = 1234
MIN_TOP3_DISTANCE = 0.10


def first_number(value) -> float:
    try:
        return float(value[0])
    except (TypeError, IndexError):
        return float(value)


def load_base_config() -> tuple[dict, Path, Path, list[str], list[str]]:
    data = yaml.safe_load(BASE_DATA.read_text(encoding="utf-8"))
    base = Path(data.get("path", BASE_DATA.parent))
    if not base.is_absolute():
        base = (BASE_DATA.parent / base).resolve()
    train_images = base / data["train"]
    val_images = base / data["val"]
    train_ids = sorted(path.stem for path in train_images.iterdir() if path.is_file())
    val_ids = sorted(path.stem for path in val_images.iterdir() if path.is_file())
    overlap = set(train_ids) & set(val_ids)
    if overlap:
        raise ValueError(f"train/val overlap contains {len(overlap)} IDs")
    if not train_ids or not val_ids:
        raise ValueError("base train/val split is empty")
    missing = [image_id for image_id in train_ids + val_ids if not (DICOM_DIR / f"{image_id}.dicom").exists()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} DICOM files; first: {missing[0]}")
    return data, base, train_images, train_ids, val_ids


def dicom_window_metadata(image_ids: list[str]) -> tuple[float, float, tuple[float, float], tuple[float, float]]:
    levels, widths = [], []
    for image_id in tqdm(image_ids, desc="read DICOM W/L metadata"):
        ds = pydicom.dcmread(DICOM_DIR / f"{image_id}.dicom", stop_before_pixels=True)
        center, width = ds.get("WindowCenter"), ds.get("WindowWidth")
        if center is None or width is None:
            continue
        center, width = first_number(center), first_number(width)
        if np.isfinite(center) and np.isfinite(width) and width > 1:
            levels.append(center)
            widths.append(width)
    if not levels:
        raise ValueError("no valid WindowCenter/WindowWidth metadata found")
    level_range = tuple(float(x) for x in np.percentile(levels, (5, 95)))
    width_range = tuple(float(x) for x in np.percentile(widths, (5, 95)))
    return float(np.median(levels)), float(np.median(widths)), level_range, width_range


def load_trace(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_trace(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def propose_candidate(
    trial: int, successful: list[dict], median_level: float, median_width: float,
    level_range: tuple[float, float], width_range: tuple[float, float], rng: random.Random,
) -> tuple[float, float, str]:
    if trial == 0:
        return median_level, median_width, "metadata_median"
    if trial < 3 or not successful:
        level = rng.uniform(*level_range)
        # Width is sampled in log space because it must stay positive.
        low, high = max(width_range[0], 2.0), max(width_range[1], width_range[0] + 2.0)
        width = math.exp(rng.uniform(math.log(low), math.log(high)))
        return level, width, "broad_random"
    best = max(successful, key=lambda row: row["best_map50"])
    progress = (trial - 3) / max(NUM_TRIALS - 3, 1)
    scale = 0.20 * (1 - progress) + 0.05 * progress
    level_span = max(level_range[1] - level_range[0], 1.0)
    level = float(best["level"]) + rng.gauss(0, level_span * scale)
    width = float(best["window"]) * math.exp(rng.gauss(0, scale))
    return level, max(width, 2.0), f"local_{scale:.3f}"


def apply_window(ds: pydicom.Dataset, level: float, width: float) -> np.ndarray:
    image = ds.pixel_array.astype(np.float32)
    image = image * float(ds.get("RescaleSlope", 1)) + float(ds.get("RescaleIntercept", 0))
    low, high = level - width / 2, level + width / 2
    image = np.clip((image - low) / max(high - low, 1e-6), 0, 1)
    if ds.get("PhotometricInterpretation") == "MONOCHROME1":
        image = 1.0 - image
    return np.rint(image * 255).astype(np.uint8)


def letterbox(image: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    new_width, new_height = round(width * scale), round(height * scale)
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    shape = (size, size) if image.ndim == 2 else (size, size, image.shape[2])
    canvas = np.zeros(shape, dtype=np.uint8)
    left, top = (size - new_width) // 2, (size - new_height) // 2
    canvas[top : top + new_height, left : left + new_width] = resized
    return canvas


def convert_one(task: tuple[str, Path, tuple[tuple[float, float], ...]]) -> tuple[str, str | None]:
    image_id, output_path, windows = task
    try:
        ds = pydicom.dcmread(DICOM_DIR / f"{image_id}.dicom")
        channels = [letterbox(apply_window(ds, level, width)) for level, width in windows]
        if len(channels) == 1:
            output = channels[0]
        elif len(channels) == 3:
            # Input is conceptual RGB; OpenCV writes BGR and the dataloader
            # converts it back to RGB, preserving the selected rank order.
            output = np.stack(channels, axis=-1)[..., ::-1]
        else:
            raise ValueError("conversion requires one or three windows")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), output):
            raise RuntimeError("cv2.imwrite returned false")
        return image_id, None
    except Exception as exc:  # returned to parent process with image context
        return image_id, f"{type(exc).__name__}: {exc}"


def ensure_labels(dataset_dir: Path, base_dir: Path) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    destination = dataset_dir / "labels"
    source = (base_dir / "labels").resolve()
    if destination.is_symlink():
        if destination.resolve() != source:
            raise ValueError(f"{destination} points to the wrong label directory")
    elif destination.exists():
        raise FileExistsError(f"{destination} exists and is not a symlink")
    else:
        destination.symlink_to(source, target_is_directory=True)


def write_data_yaml(dataset_dir: Path, base_data: dict) -> Path:
    config = {
        "path": str(dataset_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": int(base_data["nc"]),
        "names": base_data["names"],
    }
    path = dataset_dir / "data.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def build_dataset(
    dataset_dir: Path, base_dir: Path, base_data: dict, train_ids: list[str], val_ids: list[str],
    windows: tuple[tuple[float, float], ...], workers: int,
) -> Path:
    ensure_labels(dataset_dir, base_dir)
    tasks = []
    for split, image_ids in (("train", train_ids), ("val", val_ids)):
        for image_id in image_ids:
            output = dataset_dir / "images" / split / f"{image_id}.png"
            if not output.exists():
                tasks.append((image_id, output, windows))
    if tasks:
        errors = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for image_id, error in tqdm(executor.map(convert_one, tasks), total=len(tasks), desc=dataset_dir.name):
                if error:
                    errors.append((image_id, error))
        if errors:
            raise RuntimeError(f"failed to convert {len(errors)} DICOMs; first={errors[0]}")
    return write_data_yaml(dataset_dir, base_data)


def write_train_config(path: Path, data_yaml: Path, run_name: str, epochs: int, device: str | int) -> None:
    config = {
        "data": str(data_yaml), "size": MODEL_SIZE, "weights": str(WEIGHTS),
        "model_route": "image", "use_multiclass": False, "epochs": epochs,
        "image_size": IMAGE_SIZE, "batch_size": BATCH_SIZE, "workers": WORKERS,
        "device": device, "optimizer": "AdamW", "lr": 1e-3,
        "output": str(RUNS_ROOT), "name": run_name, "seed": SEED,
        "hflip": 0.5, "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.10,
        "mosaic": 0.3, "close_mosaic": min(5, epochs), "patience": epochs,
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def run_training(config_path: Path) -> None:
    code = "from train import train; import sys; train(sys.argv[1])"
    subprocess.run([sys.executable, "-c", code, str(config_path)], cwd=ROOT, check=True)


def best_result(run_dir: Path) -> tuple[float, int, dict]:
    path = run_dir / "results.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"no validation results in {path}")
    best = max(rows, key=lambda row: row["map50"])
    return float(best["map50"]), int(best["epoch"]), best


def select_diverse_top3(rows: list[dict]) -> list[dict]:
    ranked = sorted((row for row in rows if row.get("status") == "complete"),
                    key=lambda row: row["best_map50"], reverse=True)
    selected = []
    for row in ranked:
        if all(
            max(
                abs(row["window"] - old["window"]) / max(row["window"], old["window"], 1),
                abs(row["level"] - old["level"]) / max(abs(row["level"]), abs(old["level"]), 1),
            ) >= MIN_TOP3_DISTANCE
            for old in selected
        ):
            selected.append(row)
        if len(selected) == 3:
            return selected
    # A small search may not contain three diverse candidates; fill by score
    # rather than silently stopping before final training.
    selected_ids = {row["trial"] for row in selected}
    selected.extend(row for row in ranked if row["trial"] not in selected_ids)
    return selected[:3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=NUM_TRIALS)
    parser.add_argument("--search-epochs", type=int, default=SEARCH_EPOCHS)
    parser.add_argument("--final-epochs", type=int, default=FINAL_EPOCHS)
    parser.add_argument("--device", default=str(DEVICE))
    parser.add_argument("--conversion-workers", type=int, default=min(16, os.cpu_count() or 4))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.trials < 3:
        raise ValueError("at least three trials are required for a three-channel stack")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    base_data, base_dir, _, train_ids, val_ids = load_base_config()
    median_level, median_width, level_range, width_range = dicom_window_metadata(train_ids)
    print(json.dumps({
        "train_images": len(train_ids), "val_images": len(val_ids),
        "metadata_median": {"level": median_level, "window": median_width},
        "level_p05_p95": level_range, "window_p05_p95": width_range,
    }, indent=2))
    if args.dry_run:
        return

    trace_path = RUNS_ROOT / "search_results.jsonl"
    trace = load_trace(trace_path)
    rng = random.Random(SEED)
    # Advance deterministic RNG for already-recorded trials.
    for _ in range(len(trace) * 4):
        rng.random()

    completed_trials = {int(row["trial"]) for row in trace if row.get("status") == "complete"}
    for trial in range(args.trials):
        if trial in completed_trials:
            print(f"trial {trial:03d}: already complete, skipping")
            continue
        successful = [row for row in trace if row.get("status") == "complete"]
        level, window, proposal = propose_candidate(
            trial, successful, median_level, median_width, level_range, width_range, rng
        )
        trial_name = f"trial_{trial:03d}"
        dataset_dir = OUTPUT_ROOT / trial_name
        run_dir = RUNS_ROOT / trial_name
        print(f"\n{trial_name}: level={level:.3f}, window={window:.3f}, proposal={proposal}")
        try:
            data_yaml = build_dataset(
                dataset_dir, base_dir, base_data, train_ids, val_ids,
                ((level, window),), args.conversion_workers,
            )
            config_path = dataset_dir / "train.yaml"
            write_train_config(config_path, data_yaml, trial_name, args.search_epochs, args.device)
            run_training(config_path)
            map50, best_epoch, metrics = best_result(run_dir)
            record = {
                "trial": trial, "status": "complete", "proposal": proposal,
                "level": level, "window": window, "best_map50": map50,
                "best_epoch": best_epoch, "map50_95_at_best_map50": metrics["map50_95"],
                "run_dir": str(run_dir), "dataset_dir": str(dataset_dir), "seed": SEED,
            }
        except Exception as exc:
            record = {
                "trial": trial, "status": "failed", "proposal": proposal,
                "level": level, "window": window, "error": f"{type(exc).__name__}: {exc}",
            }
            append_trace(trace_path, record)
            raise
        append_trace(trace_path, record)
        trace.append(record)
        print(json.dumps(record, indent=2))

    trace = load_trace(trace_path)
    top3 = select_diverse_top3(trace)
    if len(top3) < 3:
        raise RuntimeError("fewer than three successful trials")
    (RUNS_ROOT / "top3.json").write_text(json.dumps(top3, indent=2), encoding="utf-8")
    windows = tuple((float(row["level"]), float(row["window"])) for row in top3)
    final_dir = OUTPUT_ROOT / "top3_stack"
    final_yaml = build_dataset(
        final_dir, base_dir, base_data, train_ids, val_ids, windows, args.conversion_workers
    )
    final_config = final_dir / "train.yaml"
    write_train_config(final_config, final_yaml, "top3_stack", args.final_epochs, args.device)
    print("Selected top-3 windows:")
    print(json.dumps(top3, indent=2))
    run_training(final_config)


if __name__ == "__main__":
    main()
