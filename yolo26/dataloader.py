# SPDX-License-Identifier: AGPL-3.0-only
"""Small YOLO-format image/label dataloader."""

from __future__ import annotations

import glob
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

IMAGE_SUFFIXES = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}
MASK_SUFFIXES = IMAGE_SUFFIXES | {".npy"}


def discover_images(source: str | list[str]) -> list[Path]:
    sources = source if isinstance(source, list) else [source]
    files: list[Path] = []
    for item in sources:
        p = Path(item).expanduser()
        if "*" in str(p):
            files.extend(Path(x) for x in glob.glob(str(p), recursive=True))
        elif p.is_dir():
            files.extend(x for x in p.rglob("*") if x.suffix.lower() in IMAGE_SUFFIXES)
        elif p.suffix.lower() == ".txt":
            base = p.parent
            for line in p.read_text(encoding="utf-8").splitlines():
                q = Path(line.strip())
                if line.strip():
                    files.append(q if q.is_absolute() else base / q)
        elif p.suffix.lower() in IMAGE_SUFFIXES:
            files.append(p)
        else:
            raise FileNotFoundError(f"unsupported image source: {item}")
    result = sorted({x.resolve() for x in files if x.suffix.lower() in IMAGE_SUFFIXES})
    if not result:
        raise FileNotFoundError(f"no images found in {source}")
    return result


def image_to_label_path(image: Path) -> Path:
    parts = list(image.parts)
    if "images" in parts:
        i = len(parts) - 1 - parts[::-1].index("images")
        parts[i] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image.with_suffix(".txt")


def pair_masks(images: list[Path], mask_source: str | list[str]) -> list[Path]:
    """Match one mask to every image by relative name or unique filename stem."""
    sources = mask_source if isinstance(mask_source, list) else [mask_source]
    masks = []
    for item in sources:
        path = Path(item).expanduser()
        if "*" in str(path):
            masks.extend(Path(x) for x in glob.glob(str(path), recursive=True))
        elif path.is_dir():
            masks.extend(x for x in path.rglob("*") if x.suffix.lower() in MASK_SUFFIXES)
        elif path.suffix.lower() in MASK_SUFFIXES:
            masks.append(path)
        else:
            raise FileNotFoundError(f"unsupported mask source: {item}")
    masks = sorted({x.resolve() for x in masks if x.suffix.lower() in MASK_SUFFIXES})
    if not masks:
        raise FileNotFoundError(f"no masks found in {mask_source}")
    by_stem: dict[str, list[Path]] = {}
    for path in masks:
        by_stem.setdefault(path.stem, []).append(path)

    paired = []
    for image in images:
        candidates = by_stem.get(image.stem, [])
        if not candidates:
            raise FileNotFoundError(f"no mask matching image {image.name} in {mask_source}")
        if len(candidates) > 1:
            raise ValueError(
                f"multiple masks match image stem {image.stem!r}: " + ", ".join(str(x) for x in candidates)
            )
        paired.append(candidates[0])
    return paired


def _validate_labels(labels: np.ndarray, path: Path, nc: int) -> np.ndarray:
    if (labels[:, 0] < 0).any() or (labels[:, 0] >= nc).any():
        raise ValueError(f"{path}: class id must be in [0, {nc - 1}]")
    if (labels[:, 1:] < 0).any() or (labels[:, 1:] > 1).any():
        raise ValueError(f"{path}: normalized coordinates must be in [0, 1]")
    return labels


def _read_single_class_label(path: Path, nc: int) -> np.ndarray:
    """Read the original five-column YOLO label format."""
    try:
        labels = np.loadtxt(path, dtype=np.float32, ndmin=2)
    except ValueError as e:
        raise ValueError(f"invalid label file {path}: {e}") from e
    if labels.shape[1] != 5:
        raise ValueError(f"{path}: expected `class x_center y_center width height`, got {labels.shape[1]} columns")
    if (labels[:, 0] % 1 != 0).any():
        raise ValueError(f"{path}: class IDs must be integers")
    return _validate_labels(labels, path, nc)


def _read_multiclass_label(path: Path, nc: int) -> np.ndarray:
    """Read class1 class2 ... x y w h as one multi-hot target per box."""
    if not path.exists() or path.stat().st_size == 0:
        return np.zeros((0, nc + 4), dtype=np.float32)
    targets = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            values = np.asarray([float(value) for value in line.replace(",", " ").split()], dtype=np.float32)
        except ValueError as e:
            raise ValueError(f"{path}:{line_number}: invalid numeric value") from e
        if len(values) < 5:
            raise ValueError(f"{path}:{line_number}: expected `class... x_center y_center width height`")
        classes, box = values[:-4], values[-4:]
        if (classes % 1 != 0).any():
            raise ValueError(f"{path}:{line_number}: class IDs must be integers")
        classes = classes.astype(np.int64)
        if (classes < 0).any() or (classes >= nc).any():
            raise ValueError(f"{path}:{line_number}: class id must be in [0, {nc - 1}]")
        if (box < 0).any() or (box > 1).any():
            raise ValueError(f"{path}:{line_number}: normalized coordinates must be in [0, 1]")
        multi_hot = np.zeros(nc, dtype=np.float32)
        multi_hot[np.unique(classes)] = 1.0
        targets.append(np.concatenate((multi_hot, box)))
    return np.asarray(targets, dtype=np.float32).reshape(-1, nc + 4)


def read_label(path: Path, nc: int, use_multiclass: bool = False) -> np.ndarray:
    if not path.exists() or path.stat().st_size == 0:
        columns = nc + 4 if use_multiclass else 5
        return np.zeros((0, columns), dtype=np.float32)
    if use_multiclass:
        return _read_multiclass_label(path, nc)
    else:
        return _read_single_class_label(path, nc)


def xywhn_to_xyxy(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    out = boxes.copy()
    out[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * width
    out[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * height
    out[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * width
    out[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * height
    return out


def xyxy_to_xywhn(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    out = boxes.copy()
    out[:, 0] = ((boxes[:, 0] + boxes[:, 2]) / 2) / width
    out[:, 1] = ((boxes[:, 1] + boxes[:, 3]) / 2) / height
    out[:, 2] = (boxes[:, 2] - boxes[:, 0]) / width
    out[:, 3] = (boxes[:, 3] - boxes[:, 1]) / height
    return out


def split_coarse_guide_boxes(
    classes: np.ndarray,
    boxes: np.ndarray,
    image_size: int,
    iobb_threshold: float = 0.8,
    min_area_ratio: float = 4.0,
    min_area: float = 0.10,
    min_width: float = 0.35,
    min_height: float = 0.50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Move large boxes containing a smaller same-class box to guide targets."""
    if not len(boxes):
        return classes, boxes, classes.copy(), boxes.copy()
    wh = np.maximum(boxes[:, 2:4] - boxes[:, 0:2], 0)
    areas = wh[:, 0] * wh[:, 1]
    normalized_wh = wh / float(image_size)
    normalized_area = areas / float(image_size * image_size)
    is_candidate = (
        (normalized_area >= min_area)
        | (normalized_wh[:, 0] >= min_width)
        | (normalized_wh[:, 1] >= min_height)
    )
    guide = np.zeros(len(boxes), dtype=bool)
    class_ids = classes[:, 0].astype(np.int64)
    for large_idx in np.flatnonzero(is_candidate):
        same = np.flatnonzero(class_ids == class_ids[large_idx])
        same = same[same != large_idx]
        if not len(same):
            continue
        ix1 = np.maximum(boxes[large_idx, 0], boxes[same, 0])
        iy1 = np.maximum(boxes[large_idx, 1], boxes[same, 1])
        ix2 = np.minimum(boxes[large_idx, 2], boxes[same, 2])
        iy2 = np.minimum(boxes[large_idx, 3], boxes[same, 3])
        intersection = np.maximum(ix2 - ix1, 0) * np.maximum(iy2 - iy1, 0)
        contained = intersection / np.maximum(areas[same], 1e-12) >= iobb_threshold
        much_smaller = areas[large_idx] / np.maximum(areas[same], 1e-12) >= min_area_ratio
        guide[large_idx] = bool(np.any(contained & much_smaller))
    return classes[~guide], boxes[~guide], classes[guide], boxes[guide]


def rasterize_guide(classes: np.ndarray, boxes: np.ndarray, nc: int, size: int) -> np.ndarray:
    mask = np.zeros((nc, size, size), dtype=np.float32)
    for cls, box in zip(classes, boxes):
        x1, y1, x2, y2 = np.rint(box).astype(int)
        x1, y1 = np.clip([x1, y1], 0, size)
        x2, y2 = np.clip([x2, y2], 0, size)
        if x2 > x1 and y2 > y1:
            mask[int(cls[0]), y1:y2, x1:x2] = 1.0
    return mask


def letterbox(image: np.ndarray, boxes: np.ndarray, size: int, color=(114, 114, 114)):
    """Resize without distortion and transform absolute xyxy boxes."""
    h, w = image.shape[:2]
    ratio = min(size / h, size / w)
    new_w, new_h = round(w * ratio), round(h * ratio)
    if (new_w, new_h) != (w, h):
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    left = (size - new_w) // 2
    top = (size - new_h) // 2
    image = cv2.copyMakeBorder(
        image, top, size - new_h - top, left, size - new_w - left, cv2.BORDER_CONSTANT, value=color
    )
    if len(boxes):
        boxes = boxes.copy()
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * ratio + left
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * ratio + top
    return image, boxes, (ratio, (left, top))


def letterbox_mask(mask: np.ndarray, size: int, ratio_pad) -> np.ndarray:
    """Apply an image's letterbox geometry to a mask without interpolating labels."""
    ratio, (left, top) = ratio_pad
    h, w = mask.shape[:2]
    new_w, new_h = round(w * ratio), round(h * ratio)
    if (new_w, new_h) != (w, h):
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return cv2.copyMakeBorder(
        mask, top, size - new_h - top, left, size - new_w - left, cv2.BORDER_CONSTANT, value=0
    )


def read_mask(path: Path, channels: int) -> np.ndarray:
    """Read a mask as float32 HWC, zero-padding missing feature channels."""
    if path.suffix.lower() == ".npy":
        mask = np.load(path, allow_pickle=False)
        if mask.ndim == 2:
            mask = mask[..., None]
        elif mask.ndim == 3 and mask.shape[0] <= channels and min(mask.shape[1:]) > channels:
            mask = mask.transpose(1, 2, 0)
        elif mask.ndim != 3 or mask.shape[-1] > channels or min(mask.shape[:2]) <= channels:
            raise ValueError(
                f"mask {path} must have CHW or HWC shape with at most {channels} channels, got {mask.shape}"
            )
        if mask.shape[-1] < channels:
            pad = np.zeros((*mask.shape[:2], channels - mask.shape[-1]), dtype=mask.dtype)
            mask = np.concatenate((mask, pad), axis=-1)
        elif mask.shape[-1] > channels:
            raise ValueError(f"mask {path} has {mask.shape[-1]} channels, expected at most {channels}")
        return np.ascontiguousarray(mask, dtype=np.float32)

    read_mode = cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_UNCHANGED
    mask = cv2.imread(str(path), read_mode)
    if mask is None:
        raise FileNotFoundError(f"cannot read mask: {path}")
    if mask.ndim == 2:
        mask = mask[..., None]
    if mask.shape[-1] != channels:
        raise ValueError(f"mask {path} has {mask.shape[-1]} channels, expected {channels}")
    return np.ascontiguousarray(mask, dtype=np.float32) / 255.0


def augment_hsv(image: np.ndarray, hgain: float, sgain: float, vgain: float) -> np.ndarray:
    gains = np.random.uniform(-1, 1, 3) * np.array([hgain, sgain, vgain]) + 1
    hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    x = np.arange(256, dtype=np.float32)
    lut_h = ((x * gains[0]) % 180).astype(np.uint8)
    lut_s = np.clip(x * gains[1], 0, 255).astype(np.uint8)
    lut_v = np.clip(x * gains[2], 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge((cv2.LUT(hue, lut_h), cv2.LUT(sat, lut_s), cv2.LUT(val, lut_v))), cv2.COLOR_HSV2BGR)


class YOLODataset(Dataset):
    def __init__(
        self,
        source: str | list[str],
        nc: int,
        image_size: int = 640,
        augment: bool = False,
        mosaic: float = 0.0,
        hflip: float = 0.5,
        hsv=(0.015, 0.7, 0.4),
        mask_source: str | list[str] | None = None,
        mask_channels: int = 1,
        use_multiclass: bool = False,
        use_coarse_guider: bool = False,
        guide_iobb: float = 0.8,
        guide_min_area_ratio: float = 4.0,
        guide_min_area: float = 0.10,
        guide_min_width: float = 0.35,
        guide_min_height: float = 0.50,
    ):
        self.images = discover_images(source)
        self.labels = [image_to_label_path(x) for x in self.images]
        self.masks = pair_masks(self.images, mask_source) if mask_source is not None else None
        self.nc, self.image_size, self.augment = nc, image_size, augment
        self.mosaic, self.hflip, self.hsv = mosaic, hflip, hsv
        if mask_channels < 1:
            raise ValueError("mask_channels must be positive")
        self.mask_channels = mask_channels
        self.use_multiclass = use_multiclass
        self.use_coarse_guider = use_coarse_guider
        self.guide_params = (guide_iobb, guide_min_area_ratio, guide_min_area, guide_min_width, guide_min_height)
        if use_coarse_guider and use_multiclass:
            raise ValueError("coarse guider currently supports standard single-class labels only")

    def __len__(self):
        return len(self.images)

    def _read(self, index: int, target_size: int | None = None):
        path = self.images[index]
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"cannot read image: {path}")
        h, w = image.shape[:2]
        if self.use_multiclass:
            labels = read_label(self.labels[index], self.nc, use_multiclass=True)
            classes = labels[:, : self.nc]
            normalized_boxes = labels[:, self.nc :]
        else:
            labels = read_label(self.labels[index], self.nc, use_multiclass=False)
            classes = labels[:, :1]
            normalized_boxes = labels[:, 1:5]
        boxes = xywhn_to_xyxy(normalized_boxes, w, h) if len(labels) else np.zeros((0, 4), np.float32)
        image, boxes, ratio_pad = letterbox(image, boxes, target_size or self.image_size)
        mask = None
        if self.masks is not None:
            mask = read_mask(self.masks[index], self.mask_channels)
            if mask.shape[:2] != (h, w):
                raise ValueError(
                    f"mask {self.masks[index]} has shape {mask.shape[:2]}, expected {(h, w)} for image {path}"
                )
            mask = letterbox_mask(mask, target_size or self.image_size, ratio_pad)
            if mask.ndim == 2:
                mask = mask[..., None]
        guide_classes = np.zeros((0, 1), np.float32)
        guide_boxes = np.zeros((0, 4), np.float32)
        if self.use_coarse_guider:
            classes, boxes, guide_classes, guide_boxes = split_coarse_guide_boxes(
                classes, boxes, target_size or self.image_size, *self.guide_params
            )
        return image, mask, classes, boxes, guide_classes, guide_boxes, path, (h, w), ratio_pad

    def _mosaic4(self, index: int):
        half = self.image_size // 2
        canvas = np.full((self.image_size, self.image_size, 3), 114, dtype=np.uint8)
        mask_canvas = None
        if self.masks is not None:
            shape = (self.image_size, self.image_size, self.mask_channels)
            mask_canvas = np.zeros(shape, dtype=np.float32)
        indices = [index] + random.choices(range(len(self)), k=3)
        all_cls, all_boxes, all_guide_cls, all_guide_boxes = [], [], [], []
        positions = ((0, 0), (half, 0), (0, half), (half, half))
        for idx, (x, y) in zip(indices, positions):
            image, mask, classes, boxes, guide_classes, guide_boxes, *_ = self._read(idx, half)
            canvas[y : y + half, x : x + half] = image
            if mask_canvas is not None:
                mask_canvas[y : y + half, x : x + half] = mask
            if len(boxes):
                boxes[:, [0, 2]] += x
                boxes[:, [1, 3]] += y
                all_cls.append(classes)
                all_boxes.append(boxes)
            if len(guide_boxes):
                guide_boxes[:, [0, 2]] += x
                guide_boxes[:, [1, 3]] += y
                all_guide_cls.append(guide_classes)
                all_guide_boxes.append(guide_boxes)
        class_columns = self.nc if self.use_multiclass else 1
        classes = np.concatenate(all_cls, 0) if all_cls else np.zeros((0, class_columns), np.float32)
        boxes = np.concatenate(all_boxes, 0) if all_boxes else np.zeros((0, 4), np.float32)
        guide_classes = np.concatenate(all_guide_cls, 0) if all_guide_cls else np.zeros((0, 1), np.float32)
        guide_boxes = np.concatenate(all_guide_boxes, 0) if all_guide_boxes else np.zeros((0, 4), np.float32)
        return canvas, mask_canvas, classes, boxes, guide_classes, guide_boxes, self.images[index], (self.image_size, self.image_size), (1.0, (0, 0))

    def __getitem__(self, index):
        sample = self._mosaic4(index) if self.augment and random.random() < self.mosaic else self._read(index)
        image, mask, classes, boxes, guide_classes, guide_boxes, path, original_shape, ratio_pad = sample
        if self.augment:
            image = augment_hsv(image, *self.hsv)
            if random.random() < self.hflip:
                image = np.ascontiguousarray(image[:, ::-1])
                if mask is not None:
                    mask = np.ascontiguousarray(mask[:, ::-1])
                if len(boxes):
                    x1 = boxes[:, 0].copy()
                    boxes[:, 0] = self.image_size - boxes[:, 2]
                    boxes[:, 2] = self.image_size - x1
                if len(guide_boxes):
                    x1 = guide_boxes[:, 0].copy()
                    guide_boxes[:, 0] = self.image_size - guide_boxes[:, 2]
                    guide_boxes[:, 2] = self.image_size - x1
        if len(boxes):
            boxes = boxes.clip(0, self.image_size)
            wh = boxes[:, 2:4] - boxes[:, 0:2]
            keep = (wh[:, 0] > 1) & (wh[:, 1] > 1)
            boxes, classes = boxes[keep], classes[keep]
            boxes = xyxy_to_xywhn(boxes, self.image_size, self.image_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float().div_(255)
        output = {
            "img": image,
            "cls": torch.from_numpy(classes).float(),
            "bboxes": torch.from_numpy(boxes).float(),
            "path": str(path),
            "original_shape": original_shape,
            "ratio_pad": ratio_pad,
        }
        if self.use_coarse_guider:
            output["guide_mask"] = torch.from_numpy(
                rasterize_guide(guide_classes, guide_boxes, self.nc, self.image_size)
            )
        if mask is not None:
            if mask.ndim == 2:
                mask = mask[..., None]
            mask = mask.transpose(2, 0, 1)
            output["mask"] = torch.from_numpy(np.ascontiguousarray(mask)).float()
        return output

    @staticmethod
    def collate_fn(batch):
        output = {"img": torch.stack([x["img"] for x in batch])}
        if "mask" in batch[0]:
            output["mask"] = torch.stack([x["mask"] for x in batch])
        if "guide_mask" in batch[0]:
            output["guide_mask"] = torch.stack([x["guide_mask"] for x in batch])
        output["cls"] = torch.cat([x["cls"] for x in batch], 0)
        output["bboxes"] = torch.cat([x["bboxes"] for x in batch], 0)
        output["batch_idx"] = torch.cat(
            [torch.full((len(x["cls"]),), i, dtype=torch.long) for i, x in enumerate(batch)], 0
        )
        for key in ("path", "original_shape", "ratio_pad"):
            output[key] = [x[key] for x in batch]
        return output


def create_dataloader(
    source: str,
    nc: int,
    image_size=640,
    batch_size=16,
    workers=8,
    augment=False,
    shuffle=None,
    mosaic=0.0,
    hflip=0.5,
    hsv=(0.015, 0.7, 0.4),
    mask_source=None,
    mask_channels=1,
    distributed=False,
    rank=0,
    world_size=1,
    use_multiclass=False,
    use_coarse_guider=False,
    guide_iobb=0.8,
    guide_min_area_ratio=4.0,
    guide_min_area=0.10,
    guide_min_width=0.35,
    guide_min_height=0.50,
):
    if use_multiclass:
        dataset = YOLODataset(
            source, nc, image_size, augment, mosaic, hflip, hsv, mask_source, mask_channels,
            use_multiclass=True, use_coarse_guider=use_coarse_guider,
            guide_iobb=guide_iobb, guide_min_area_ratio=guide_min_area_ratio,
            guide_min_area=guide_min_area, guide_min_width=guide_min_width, guide_min_height=guide_min_height,
        )
    else:
        dataset = YOLODataset(
            source, nc, image_size, augment, mosaic, hflip, hsv, mask_source, mask_channels,
            use_multiclass=False, use_coarse_guider=use_coarse_guider,
            guide_iobb=guide_iobb, guide_min_area_ratio=guide_min_area_ratio,
            guide_min_area=guide_min_area, guide_min_width=guide_min_width, guide_min_height=guide_min_height,
        )
    sampler = (
        DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=augment)
        if distributed
        else None
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False if sampler is not None else (augment if shuffle is None else shuffle),
        sampler=sampler,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        collate_fn=dataset.collate_fn,
        drop_last=False,
    )
