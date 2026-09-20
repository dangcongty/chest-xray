# SPDX-License-Identifier: AGPL-3.0-only
"""MuSGD and ordinary optimizer builders."""

from __future__ import annotations

import torch
from torch import nn


def zeropower_via_newtonschulz5(g: torch.Tensor, eps=1e-7):
    x = g.flatten(1).bfloat16()
    x /= x.norm() + eps
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(5):
        aa = x @ x.T
        x = a * x + (b * aa + c * aa @ aa) @ x
    if transposed:
        x = x.T
    scale = max(1, g.shape[-2] / g.shape[-1]) ** 0.5
    return x.reshape_as(g).to(g.dtype) * scale


class MuSGD(torch.optim.Optimizer):
    """Hybrid Muon + SGD optimizer used by the public YOLO26 training pipeline."""

    def __init__(self, params, lr=1e-3, momentum=0.9, weight_decay=0.0, nesterov=True, muon=0.2, sgd=1.0):
        super().__init__(
            params,
            {"lr": lr, "momentum": momentum, "weight_decay": weight_decay, "nesterov": nesterov, "use_muon": False},
        )
        self.muon, self.sgd = muon, sgd

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, beta, decay = group["lr"], group["momentum"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                grad = p.grad
                if group.get("use_muon", False):
                    mu_buf = state.setdefault("muon_buffer", torch.zeros_like(p))
                    mu_buf.mul_(beta).add_(grad, alpha=1 - beta)
                    update = mu_buf.mul(beta).add(grad, alpha=1 - beta) if group["nesterov"] else mu_buf
                    p.add_(zeropower_via_newtonschulz5(update), alpha=-(lr * self.muon))
                    sgd_lr = lr * self.sgd
                else:
                    sgd_lr = lr
                if decay:
                    grad = grad.add(p, alpha=decay)
                buf = state.setdefault("momentum_buffer", torch.zeros_like(p))
                buf.mul_(beta).add_(grad)
                update = grad.add(buf, alpha=beta) if group["nesterov"] else buf
                p.add_(update, alpha=-sgd_lr)
        return loss


def build_optimizer(model: nn.Module, name: str, lr: float, momentum: float, weight_decay: float):
    name = name.lower()
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, betas=(momentum, 0.999), weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=True, weight_decay=weight_decay)
    if name != "musgd":
        raise ValueError("optimizer must be AdamW, SGD or MuSGD")
    boosted = {id(p) for branch in (model.head.cv3, model.head.one2one_cv3) for p in branch.parameters()}
    muon, muon_boosted, regular, regular_boosted = [], [], [], []
    for _, module in model.named_modules():
        for _, p in module.named_parameters(recurse=False):
            if p.ndim in {2, 4}:
                (muon_boosted if id(p) in boosted else muon).append(p)
            else:
                (regular_boosted if id(p) in boosted else regular).append(p)
    groups = [
        {"params": muon, "use_muon": True, "weight_decay": weight_decay, "lr_scale": 1.0},
        {"params": muon_boosted, "use_muon": True, "weight_decay": weight_decay, "lr": lr * 3, "lr_scale": 3.0},
        {"params": regular, "use_muon": False, "weight_decay": 0.0, "lr_scale": 1.0},
        {"params": regular_boosted, "use_muon": False, "weight_decay": 0.0, "lr": lr * 3, "lr_scale": 3.0},
    ]
    return MuSGD(groups, lr=lr, momentum=momentum, nesterov=True)
