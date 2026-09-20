"""Generate four-channel anatomical masks for the mask-guided YOLO26 model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pydicom
import torch
import torchxrayvision as xrv
from tqdm import tqdm

TARGET_INDICES = (4, 5, 8, 9)  # left lung, right lung, heart, aorta
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
COLORS = ((255, 0, 255), (0, 255, 255), (0, 0, 128), (128, 128, 0))


def first_number(value) -> float:
    try:
        return float(value[0])
    except (TypeError, IndexError):
        return float(value)


def read_dicom(path: Path) -> np.ndarray:
    """Decode DICOM identically to tools/dicom2png.py."""
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
        raise ValueError(f"invalid DICOM pixel range [{low}, {high}]")
    image = np.clip((np.clip(image, low, high) - low) * (255.0 / (high - low)), 0, 255)
    if dataset.get("PhotometricInterpretation") == "MONOCHROME1":
        image = 255.0 - image
    return image.astype(np.uint8)


def letterbox_black(image: np.ndarray, size: int = 640) -> np.ndarray:
    """Match the black-padded square images used during training."""
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    new_width, new_height = round(width * scale), round(height * scale)
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    left, top = (size - new_width) // 2, (size - new_height) // 2
    canvas = np.zeros((size, size), dtype=np.uint8)
    canvas[top : top + new_height, left : left + new_width] = resized
    return canvas


def read_image(path: Path, image_size: int) -> np.ndarray:
    if path.suffix.lower() == ".dicom":
        return letterbox_black(read_dicom(path), image_size)
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return image if image.shape == (image_size, image_size) else letterbox_black(image, image_size)


def discover_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    paths = [
        path for path in source.rglob("*")
        if path.is_file() and (path.suffix.lower() in IMAGE_SUFFIXES or path.suffix.lower() == ".dicom")
    ]
    if not paths:
        raise FileNotFoundError(f"No supported images found in {source}")
    return sorted(paths)


def ensure_probs(output: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(output) if output.min() < 0 or output.max() > 1 else output


def save_visualization(path: Path, image: np.ndarray, masks: np.ndarray, names: list[str]) -> None:
    overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for channel, (mask, color, name) in enumerate(zip(masks, COLORS, names)):
        selected = mask.astype(bool)
        overlay[selected] = (
            overlay[selected].astype(np.float32) * 0.65 + np.asarray(color, dtype=np.float32) * 0.35
        ).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 1)
        cv2.putText(overlay, name, (8, 22 + channel * 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), overlay)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate four-channel chest anatomy masks")
    parser.add_argument("--input", type=Path, default=Path("datasets/png/images"))
    parser.add_argument("--output", type=Path, default=Path("datasets/png/seg"))
    parser.add_argument("--visualize", type=Path, default=None,
                        help="Optional directory for mask overlay PNGs")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    paths = discover_images(args.input)
    args.output.mkdir(parents=True, exist_ok=True)
    model = xrv.baseline_models.chestx_det.PSPNet().eval().to(args.device)
    target_names = [model.targets[index] for index in TARGET_INDICES]
    resizer = xrv.datasets.XRayResizer(512)
    completed, skipped, failed = 0, 0, []

    for path in tqdm(paths, desc="segment"):
        output_path = args.output / f"{path.stem}.npy"
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            image = read_image(path, args.image_size)
            normalized = xrv.datasets.normalize(image, 255)[None, ...]
            tensor = torch.from_numpy(resizer(normalized)).float().unsqueeze(0).to(args.device)
            probabilities = ensure_probs(model(tensor)[0].cpu())
            # Never drop an empty channel: channel position is part of the model input contract.
            masks = np.stack([
                cv2.resize(probabilities[index].numpy(), (args.image_size, args.image_size),
                           interpolation=cv2.INTER_LINEAR) > args.threshold
                for index in TARGET_INDICES
            ]).astype(np.uint8)
            np.save(output_path, masks)
            if args.visualize is not None:
                save_visualization(args.visualize / f"{path.stem}.png", image, masks, target_names)
            completed += 1
        except Exception as exc:
            failed.append((str(path), str(exc)))

    print(f"Saved: {completed} | skipped: {skipped} | failed: {len(failed)}")
    if failed:
        failure_path = args.output / "failed.json"
        failure_path.write_text(json.dumps(failed, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Failure details: {failure_path}")


if __name__ == "__main__":
    main()
