# SPDX-License-Identifier: AGPL-3.0-only
"""Single-GPU YOLO26 detector trainer."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from config import TrainConfig, load_data_config, merge_config
from dataloader import create_dataloader
from losses import YOLO26Loss
from model import build_model
from optimizer import build_optimizer
from utils import AverageMeter, ModelEMA, append_jsonl, forward_batch, move_batch, seed_everything, select_device
from val import evaluate


def cosine_factor(epoch, epochs, final_ratio):
    return ((1 + math.cos(math.pi * epoch / epochs)) / 2) * (1 - final_ratio) + final_ratio


def save_checkpoint(path, model, ema, optimizer, epoch, best, cfg, names):
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


def train(cfg: TrainConfig):
    seed_everything(cfg.seed)
    data = load_data_config(cfg.data)
    if cfg.model_route == "mask_guider":
        missing = [key for key in ("train_masks", "val_masks") if not data.get(key)]
        if missing:
            raise ValueError(f"mask_guider route requires {', '.join(missing)} in the dataset YAML")
    device = select_device(cfg.device)
    run_dir = Path(cfg.output) / cfg.name
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8")

    model = build_model(
        size=cfg.size, nc=data["nc"], model_route=cfg.model_route, mask_channels=cfg.mask_channels
    ).to(device)
    if cfg.weights:
        result = model.load_compact(cfg.weights, strict=False)
        print(
            f"Loaded pretrained weights: missing={len(result.missing_keys)}, unexpected={len(result.unexpected_keys)}"
        )
    optimizer = build_optimizer(model, cfg.optimizer, cfg.lr, cfg.momentum, cfg.weight_decay)
    ema = ModelEMA(model) if cfg.ema else None
    start_epoch, best, stale = 0, -1.0, 0
    if cfg.resume:
        checkpoint = torch.load(cfg.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
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
        use_multiclass=cfg.use_multiclass,
        use_coarse_guider=cfg.model_route == "coarse_guider",
        guide_iobb=cfg.guide_iobb,
        guide_min_area_ratio=cfg.guide_min_area_ratio,
        guide_min_area=cfg.guide_min_area,
        guide_min_width=cfg.guide_min_width,
        guide_min_height=cfg.guide_min_height,
    )
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
        criterion = YOLO26Loss(model, cfg.epochs, cfg.box, cfg.cls, cfg.l1, use_multiclass=True,
                               guide_loss_weight=cfg.guide_loss_weight)
    else:
        criterion = YOLO26Loss(model, cfg.epochs, cfg.box, cfg.cls, cfg.l1, use_multiclass=False,
                               guide_loss_weight=cfg.guide_loss_weight)
    amp_enabled = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    print(
        f"YOLO26{cfg.size}: {model.num_parameters():,} parameters | device={device} | images={len(train_loader.dataset)}"
    )

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        criterion.set_epoch(epoch)
        train_loader.dataset.mosaic = 0.0 if epoch >= cfg.epochs - cfg.close_mosaic else cfg.mosaic
        lr_factor = cosine_factor(epoch, cfg.epochs, cfg.final_lr_ratio)
        for group in optimizer.param_groups:
            group["lr"] = cfg.lr * lr_factor * group.get("lr_scale", 1.0)
        loss_meter = AverageMeter()
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{cfg.epochs}")
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
                loss_vec, items = criterion(forward_batch(model, batch), batch)
                loss = loss_vec.sum()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(optimizer)
            scaler.update()
            if ema:
                ema.update(model)
            loss_meter.update(loss.item(), batch["img"].shape[0])
            progress.set_postfix(
                loss=f"{loss_meter.avg:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                o2m=f"{items['o2m_weight']:.2f}",
            )

        eval_model = ema.ema if ema else model
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
        record = {"epoch": epoch + 1, "train_loss": loss_meter.avg, "lr": optimizer.param_groups[0]["lr"], **val_result}
        append_jsonl(run_dir / "results.jsonl", record)
        print(json.dumps({k: v for k, v in record.items() if k != "per_class"}, ensure_ascii=False))
        fitness = float(val_result["map50_95"])
        save_checkpoint(weights_dir / "last.pt", model, ema, optimizer, epoch, max(best, fitness), cfg, data["names"])
        if fitness > best:
            best, stale = fitness, 0
            save_checkpoint(weights_dir / "best.pt", model, ema, optimizer, epoch, best, cfg, data["names"])
        else:
            stale += 1
        if cfg.save_period > 0 and (epoch + 1) % cfg.save_period == 0:
            save_checkpoint(
                weights_dir / f"epoch{epoch + 1}.pt", model, ema, optimizer, epoch, best, cfg, data["names"]
            )
        if stale >= cfg.patience:
            print(f"Early stopping: no mAP50-95 improvement for {cfg.patience} epochs")
            break
    return run_dir


def parse_args():
    p = argparse.ArgumentParser(description="Train compact YOLO26 detection model")
    p.add_argument("--config", help="optional training YAML")
    p.add_argument("--data")
    p.add_argument("--size", choices=list("nsmxl"))
    p.add_argument("--weights")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--image-size", type=int)
    p.add_argument("--workers", type=int)
    p.add_argument("--device")
    p.add_argument("--optimizer", choices=["AdamW", "SGD", "MuSGD"])
    p.add_argument("--lr", type=float)
    p.add_argument("--output")
    p.add_argument("--name")
    p.add_argument("--resume")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--use-multiclass", action="store_true", default=None)
    args = vars(p.parse_args())
    config_path = args.pop("config")
    no_amp = args.pop("no_amp")
    if no_amp:
        args["amp"] = False
    cfg = merge_config(config_path, args)
    if not cfg.data:
        p.error("--data or `data:` in --config is required")
    return cfg


if __name__ == "__main__":
    train(parse_args())
