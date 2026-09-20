# SPDX-License-Identifier: AGPL-3.0-only
"""Runtime helpers kept separate from model/data/loss code."""

from __future__ import annotations

import copy
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(spec=""):
    if spec == "cpu":
        return torch.device("cpu")
    if isinstance(spec, list):
        if len(spec) != 1:
            raise ValueError("multiple devices must be launched through train(device=[...]) or torchrun")
        spec = spec[0]
    if isinstance(spec, int):
        spec = str(spec)
    if spec:
        return torch.device(spec if spec.startswith("cuda") else f"cuda:{spec}")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def setup_distributed(spec=""):
    """Initialize torchrun DDP and return device/rank metadata."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size == 1:
        return select_device(spec), rank, world_size, local_rank
    if not torch.cuda.is_available():
        raise RuntimeError("multi-GPU DDP requires CUDA")
    device_ids = spec if isinstance(spec, list) else list(range(world_size))
    if len(device_ids) < world_size:
        raise RuntimeError(f"DDP world size is {world_size}, but device only contains {len(device_ids)} GPU IDs")
    device_id = int(device_ids[local_rank])
    if device_id >= torch.cuda.device_count():
        raise RuntimeError(f"GPU {device_id} is unavailable; visible GPU count is {torch.cuda.device_count()}")
    torch.cuda.set_device(device_id)
    dist.init_process_group(backend="nccl", init_method="env://", device_id=torch.device("cuda", device_id))
    return torch.device("cuda", device_id), rank, world_size, device_id


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def move_batch(batch, device):
    for key in ("img", "mask", "cls", "bboxes", "batch_idx"):
        if key not in batch:
            continue
        batch[key] = batch[key].to(device, non_blocking=True)
    return batch


def forward_batch(model, batch):
    """Route a batch through either the image-only or mask-guided model."""
    core_model = model.module if hasattr(model, "module") else model
    if getattr(core_model, "requires_mask", False):
        if "mask" not in batch:
            raise KeyError("mask-guided model requires batch['mask']")
        return model(batch["img"], batch["mask"])
    return model(batch["img"])


class ModelEMA:
    def __init__(self, model, decay=0.9999, tau=2000):
        self.ema = copy.deepcopy(model).eval()
        self.updates, self.decay, self.tau = 0, decay, tau
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / self.tau))
        source = model.state_dict()
        for key, value in self.ema.state_dict().items():
            if value.dtype.is_floating_point:
                value.mul_(d).add_(source[key].detach(), alpha=1 - d)


class AverageMeter:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, value, n=1):
        self.total += float(value) * n
        self.count += n

    @property
    def avg(self):
        return self.total / max(self.count, 1)


def append_jsonl(path: str | Path, record: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
