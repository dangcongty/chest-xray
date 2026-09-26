# SPDX-License-Identifier: AGPL-3.0-only
"""Configuration shared by train/validation/inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

MODEL_SCALES: dict[str, tuple[float, float, int]] = {
    "n": (0.50, 0.25, 1024),
    "s": (0.50, 0.50, 1024),
    "m": (0.50, 1.00, 512),
    "l": (1.00, 1.00, 512),
    "x": (1.00, 1.50, 512),
}


@dataclass
class TrainConfig:
    data: str = ""
    size: str = "n"
    weights: str | None = None
    epochs: int = 100
    batch_size: int = 16
    image_size: int = 640
    workers: int = 8
    device: str | int | list[int] = ""
    optimizer: str = "AdamW"  # AdamW, SGD or MuSGD
    lr: float = 1e-3
    final_lr_ratio: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 5e-4
    warmup_epochs: float = 3.0
    box: float = 7.5
    cls: float = 0.5
    l1: float = 1.5  # Ultralytics keeps the historical CLI name `dfl`; YOLO26 uses L1 here.
    mosaic: float = 1.0
    hflip: float = 0.5
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    close_mosaic: int = 10
    patience: int = 50
    seed: int = 0
    amp: bool = True
    ema: bool = True
    output: str = "runs/train"
    name: str = "exp"
    resume: str | None = None
    save_period: int = -1
    conf: float = 0.001
    iou: float = 0.7
    max_det: int = 300

    # Model/data route. ``image`` keeps the original YOLO26 backbone.
    # ``mask_guider`` uses external masks. ``coarse_guider`` learns coarse
    # class-aware regions from large boxes and uses them to guide small boxes.
    model_route: str = "mask_guider"
    mask_channels: int = 4
    guide_loss_weight: float = 0.5
    guide_iobb: float = 0.8
    guide_min_area_ratio: float = 4.0
    guide_min_area: float = 0.10
    guide_min_width: float = 0.35
    guide_min_height: float = 0.50
    # Label route. False keeps standard YOLO rows: class x y w h.
    # True enables multi-class rows: class1 class2 ... x y w h.
    use_multiclass: bool = False

    def validate(self) -> None:
        if self.size not in MODEL_SCALES:
            raise ValueError(f"size must be one of {tuple(MODEL_SCALES)}, got {self.size!r}")
        if self.image_size % 32:
            raise ValueError("image_size must be divisible by 32")
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if self.model_route not in {"image", "mask_guider", "coarse_guider", "stn"}:
            raise ValueError("model_route must be 'image', 'mask_guider', 'coarse_guider' or 'stn'")
        if self.mask_channels < 1:
            raise ValueError("mask_channels must be positive")
        if not isinstance(self.use_multiclass, bool):
            raise ValueError("use_multiclass must be a bool")
        if self.model_route == "coarse_guider" and self.use_multiclass:
            raise ValueError("coarse_guider currently requires standard single-class YOLO labels")
        if self.guide_loss_weight < 0:
            raise ValueError("guide_loss_weight must be non-negative")
        if not 0 <= self.guide_iobb <= 1:
            raise ValueError("guide_iobb must be in [0, 1]")
        if self.guide_min_area_ratio <= 1:
            raise ValueError("guide_min_area_ratio must be > 1")
        if isinstance(self.device, list):
            if not self.device:
                raise ValueError("device list cannot be empty")
            if any(not isinstance(x, int) or x < 0 for x in self.device):
                raise ValueError("device list must contain non-negative GPU indices")
            if len(set(self.device)) != len(self.device):
                raise ValueError("device list cannot contain duplicate GPU indices")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    data["yaml_file"] = str(path)
    return data


def load_data_config(path: str | Path) -> dict[str, Any]:
    """Load Ultralytics-style dataset YAML and resolve relative paths."""
    data = load_yaml(path)
    root = Path(data.get("path") or Path(data["yaml_file"]).parent).expanduser()
    if not root.is_absolute():
        root = (Path(data["yaml_file"]).parent / root).resolve()
    for key in ("train", "val", "test", "train_masks", "val_masks", "test_masks"):
        if key in data and isinstance(data[key], str):
            p = Path(data[key]).expanduser()
            data[key] = str(p if p.is_absolute() else root / p)
    names = data.get("names")
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=lambda x: int(x))]
    if not names:
        nc = int(data.get("nc", 0))
        names = [str(i) for i in range(nc)]
    data["names"] = names
    data["nc"] = len(names)
    if not data["nc"]:
        raise ValueError("dataset YAML must define non-empty `names` or `nc`")
    return data


def merge_config(yaml_path: str | None, overrides: dict[str, Any]) -> TrainConfig:
    values: dict[str, Any] = {}
    if yaml_path:
        values.update(load_yaml(yaml_path))
        values.pop("yaml_file", None)
    values.update({k: v for k, v in overrides.items() if v is not None})
    cfg = TrainConfig(**values)
    cfg.validate()
    return cfg
