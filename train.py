# SPDX-License-Identifier: AGPL-3.0-only
"""Single- and multi-GPU (torchrun DDP) YOLO26 detector trainer."""

from __future__ import annotations

import json
import math
import os
import socket
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from yolo26.config import TrainConfig, load_data_config, merge_config
from yolo26.dataloader import create_dataloader
from yolo26.losses import YOLO26Loss
from yolo26.model import YOLO26, build_model, stn_targets
from yolo26.optimizer import build_optimizer
from yolo26.utils import (
    AverageMeter,
    ModelEMA,
    append_jsonl,
    cleanup_distributed,
    forward_batch,
    move_batch,
    seed_everything,
    setup_distributed,
)
from yolo26.val import evaluate


def cosine_factor(epoch, epochs, final_ratio):
    return ((1 + math.cos(math.pi * epoch / epochs)) / 2) * (1 - final_ratio) + final_ratio


def save_checkpoint(path, model:YOLO26, ema:ModelEMA, optimizer, epoch, best, cfg, names):
    checkpoint = {
        "model": model.state_dict(),
        "ema": ema.ema.state_dict() if ema else None,
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_fitness": best,
        "size": model.size,
        "nc": model.nc,
        "names": names,
        "model_route": model.model_route,
        "mask_channels": getattr(model, "mask_channels", None),
        "use_multiclass": cfg.use_multiclass,
        "train_args": cfg.to_dict(),
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    torch.save(checkpoint, path)


def _train(cfg: TrainConfig):
    device, rank, world_size, local_rank = setup_distributed(cfg.device)
    distributed = world_size > 1
    is_main = rank == 0
    seed_everything(cfg.seed + rank)
    data = load_data_config(cfg.data)
    if cfg.model_route == "mask_guider":
        missing = [key for key in ("train_masks", "val_masks") if not data.get(key)]
        if missing:
            raise ValueError(f"mask_guider route requires {', '.join(missing)} in the dataset YAML")
    run_dir = Path(cfg.output) / cfg.name
    weights_dir = run_dir / "weights"
    if is_main:
        weights_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "args.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8")
    if distributed:
        dist.barrier()

    model_core = build_model(
        size=cfg.size, nc=data["nc"], model_route=cfg.model_route, mask_channels=cfg.mask_channels
    ).to(device)
    if cfg.weights:
        result = model_core.load_compact(cfg.weights, strict=False)
        if is_main:
            print(
                f"Loaded pretrained weights: missing={len(result.missing_keys)}, unexpected={len(result.unexpected_keys)}"
            )
    model = DDP(model_core, device_ids=[local_rank], output_device=local_rank) if distributed else model_core
    optimizer = build_optimizer(model_core, cfg.optimizer, cfg.lr, cfg.momentum, cfg.weight_decay)
    ema = ModelEMA(model_core) if cfg.ema and is_main else None
    start_epoch, best, stale = 0, -1.0, 0
    if cfg.resume:
        checkpoint = torch.load(cfg.resume, map_location=device, weights_only=False)
        model_core.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best = float(checkpoint.get("best_fitness", -1))
        if ema and checkpoint.get("ema"):
            ema.ema.load_state_dict(checkpoint["ema"])

    train_loader = create_dataloader(
        data["train"],
        data["nc"],
        cfg.image_size,
        cfg.batch_size,
        cfg.workers,
        True,
        mosaic=cfg.mosaic,
        hflip=cfg.hflip,
        hsv=(cfg.hsv_h, cfg.hsv_s, cfg.hsv_v),
        mask_source=data.get("train_masks") if cfg.model_route == "mask_guider" else None,
        mask_channels=cfg.mask_channels,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        use_multiclass=cfg.use_multiclass,
        use_coarse_guider=cfg.model_route == "coarse_guider",
        guide_iobb=cfg.guide_iobb,
        guide_min_area_ratio=cfg.guide_min_area_ratio,
        guide_min_area=cfg.guide_min_area,
        guide_min_width=cfg.guide_min_width,
        guide_min_height=cfg.guide_min_height,
    )
    val_loader = None
    if is_main:
        val_loader = create_dataloader(
            data["val"], data["nc"], cfg.image_size, cfg.batch_size, cfg.workers,
            mask_source=data.get("val_masks") if cfg.model_route == "mask_guider" else None,
            mask_channels=cfg.mask_channels,
            use_multiclass=cfg.use_multiclass,
            use_coarse_guider=cfg.model_route == "coarse_guider",
            guide_iobb=cfg.guide_iobb,
            guide_min_area_ratio=cfg.guide_min_area_ratio,
            guide_min_area=cfg.guide_min_area,
            guide_min_width=cfg.guide_min_width,
            guide_min_height=cfg.guide_min_height,
        )
    if cfg.use_multiclass:
        criterion = YOLO26Loss(model_core, cfg.epochs, cfg.box, cfg.cls, cfg.l1, use_multiclass=True,
                               guide_loss_weight=cfg.guide_loss_weight)
    else:
        criterion = YOLO26Loss(model_core, cfg.epochs, cfg.box, cfg.cls, cfg.l1, use_multiclass=False,
                               guide_loss_weight=cfg.guide_loss_weight)
    amp_enabled = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    if is_main:
        print(
            f"YOLO26{cfg.size}: {model_core.num_parameters():,} parameters | "
            f"device={device} | world_size={world_size} | images={len(train_loader.dataset)}"
        )

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        if distributed:
            train_loader.sampler.set_epoch(epoch)
        criterion.set_epoch(epoch)
        train_loader.dataset.mosaic = 0.0 if epoch >= cfg.epochs - cfg.close_mosaic else cfg.mosaic
        lr_factor = cosine_factor(epoch, cfg.epochs, cfg.final_lr_ratio)
        for group in optimizer.param_groups:
            group["lr"] = cfg.lr * lr_factor * group.get("lr_scale", 1.0)
        loss_meter = AverageMeter()
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{cfg.epochs}", disable=not is_main)
        for step, batch in enumerate(progress):
            batch = move_batch(batch, device)
            # Linear warmup adjusts LR only; momentum stays stable for a compact, predictable implementation.
            warmup_steps = max(round(cfg.warmup_epochs * len(train_loader)), 1)
            global_step = epoch * len(train_loader) + step
            if global_step < warmup_steps:
                warm = (global_step + 1) / warmup_steps
                for group in optimizer.param_groups:
                    group["lr"] = cfg.lr * lr_factor * warm * group.get("lr_scale", 1.0)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                raw = forward_batch(model, batch)
                targets = stn_targets(batch, raw["stn_theta"]) if cfg.model_route == "stn" else batch
                loss_vec, items = criterion(raw, targets)
                loss = loss_vec.sum()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(optimizer)
            scaler.update()
            if ema:
                ema.update(model_core)
            loss_meter.update(loss.item(), batch["img"].shape[0])
            if is_main:
                progress.set_postfix(
                    loss=f"{loss_meter.avg:.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                    o2m=f"{items['o2m_weight']:.2f}",
                    guide=f"{items['guide']:.3f}" if "guide" in items else "-",
                )

        totals = torch.tensor([loss_meter.total, loss_meter.count], device=device, dtype=torch.float64)
        if distributed:
            dist.reduce(totals, dst=0, op=dist.ReduceOp.SUM)

        should_stop = False
        if is_main:
            train_loss = (totals[0] / totals[1].clamp_min(1)).item()
            eval_model = ema.ema if ema else model_core
            val_result = evaluate(
                eval_model,
                val_loader,
                device,
                data["names"],
                cfg.image_size,
                cfg.conf,
                cfg.iou,
                cfg.max_det,
                end2end=False,
                criterion=None,
                use_multiclass=cfg.use_multiclass,
            )
            record = {"epoch": epoch + 1, "train_loss": train_loss, "lr": optimizer.param_groups[0]["lr"], **val_result}
            append_jsonl(run_dir / "results.jsonl", record)
            print(json.dumps({k: v for k, v in record.items() if k != "per_class"}, ensure_ascii=False))
            fitness = float(val_result["map50_95"])
            save_checkpoint(weights_dir / "last.pt", model_core, ema, optimizer, epoch, max(best, fitness), cfg, data["names"])
            if fitness > best:
                best, stale = fitness, 0
                save_checkpoint(weights_dir / "best.pt", model_core, ema, optimizer, epoch, best, cfg, data["names"])
            else:
                stale += 1
            if cfg.save_period > 0 and (epoch + 1) % cfg.save_period == 0:
                save_checkpoint(
                    weights_dir / f"epoch{epoch + 1}.pt", model_core, ema, optimizer, epoch, best, cfg, data["names"]
                )
            should_stop = stale >= cfg.patience
            if should_stop:
                print(f"Early stopping: no mAP50-95 improvement for {cfg.patience} epochs")

        if distributed:
            stop_tensor = torch.tensor(int(should_stop), device=device)
            dist.broadcast(stop_tensor, src=0)
            should_stop = bool(stop_tensor.item())
        if should_stop:
            break
    cleanup_distributed()
    return run_dir


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _spawn_worker(local_rank: int, world_size: int, config_values: dict, master_port: int):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    try:
        _train(TrainConfig(**config_values))
    finally:
        cleanup_distributed()


def train(config: TrainConfig | dict | str | Path | None = None, **kwargs):
    """Train YOLO26 using direct keyword arguments, a config object/dict, or YAML.

    Examples:
        train(data="data.yaml", epochs=100, batch_size=16)
        train("configs/train_n.yaml", epochs=200)
        train(TrainConfig(data="data.yaml"))

    A few familiar Ultralytics argument aliases are accepted: ``imgsz``,
    ``batch``, ``lr0``, ``lrf``, ``fliplr`` and ``project``.
    """
    aliases = {
        "imgsz": "image_size",
        "batch": "batch_size",
        "lr0": "lr",
        "lrf": "final_lr_ratio",
        "fliplr": "hflip",
        "project": "output",
    }
    normalized = {}
    for key, value in kwargs.items():
        target = aliases.get(key, key)
        if target in normalized:
            raise ValueError(f"duplicate training argument for {target!r}")
        normalized[target] = value

    if config is None:
        cfg = TrainConfig(**normalized)
    elif isinstance(config, TrainConfig):
        values = config.to_dict()
        values.update(normalized)
        cfg = TrainConfig(**values)
    elif isinstance(config, dict):
        values = {**config, **normalized}
        cfg = TrainConfig(**values)
    elif isinstance(config, (str, Path)):
        cfg = merge_config(str(config), normalized)
    else:
        raise TypeError("config must be TrainConfig, dict, YAML path, or None")

    cfg.validate()
    if not cfg.data:
        raise ValueError("data is required")

    # Match the Ultralytics-style API: a GPU list automatically launches one
    # DDP process per selected device. Under torchrun, process creation is
    # already handled externally, so each rank enters _train directly.
    if isinstance(cfg.device, list) and len(cfg.device) > 1 and int(os.environ.get("WORLD_SIZE", "1")) == 1:
        if not torch.cuda.is_available():
            raise RuntimeError("device=[...] requires CUDA")
        unavailable = [gpu for gpu in cfg.device if gpu >= torch.cuda.device_count()]
        if unavailable:
            raise RuntimeError(
                f"GPU IDs {unavailable} are unavailable; visible GPU count is {torch.cuda.device_count()}"
            )
        world_size = len(cfg.device)
        mp.spawn(
            _spawn_worker,
            args=(world_size, cfg.to_dict(), _free_port()),
            nprocs=world_size,
            join=True,
        )
        return Path(cfg.output) / cfg.name
    return _train(cfg)


if __name__ == "__main__":
    # One GPU: device="0" or device=0
    # Multiple GPUs: device=[0, 1] (DDP processes are launched automatically)
    train(
        data="/mnt/workspace/ty/xray/datasets/yolo/data.yaml",
        size="m",
        weights="yolo26m.pt",
        model_route="stn",  # image | mask_guider | coarse_guider | stn
        mask_channels=4,
        guide_loss_weight=0.5,
        guide_iobb=0.8,
        guide_min_area_ratio=4.0,
        use_multiclass=False,
        epochs=500,
        image_size=640,
        batch_size=16,
        workers=8,
        device=0,
        optimizer="AdamW",
        lr=1e-3,
        name="stn-yolo-attempt-1",
        seed=1234,
        hflip=0.5,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.10,
        mosaic=0.3,
        close_mosaic=20,
    )
