# SPDX-License-Identifier: AGPL-3.0-only
"""Compact detection-only YOLO26 architecture.

The module layout intentionally mirrors Ultralytics YOLO26 so converted official
weights can be loaded by key. Only layers used by the P3/P4/P5 detection model are
included.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import torch
from torch import nn
from torchvision.ops import nms

try:
    from .config import MODEL_SCALES
except ImportError:  # Support running scripts directly from the yolo26 directory.
    from config import MODEL_SCALES


def make_divisible(x: float, divisor: int = 8) -> int:
    return math.ceil(x / divisor) * divisor


def autopad(k: int | tuple[int, int], p=None, d: int = 1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else tuple(d * (x - 1) + 1 for x in k)
    return (k // 2 if isinstance(k, int) else tuple(x // 2 for x in k)) if p is None else p


class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DWConv(Conv):
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3k(C3):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        self.qkv = Conv(dim, dim + nh_kd * 2, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        b, c, h, w = x.shape
        n = h * w
        q, k, v = (
            self.qkv(x)
            .view(b, self.num_heads, self.key_dim * 2 + self.head_dim, n)
            .split([self.key_dim, self.key_dim, self.head_dim], dim=2)
        )
        attn = ((q * self.scale).transpose(-2, -1) @ k).softmax(dim=-1)
        y = (v @ attn.transpose(-2, -1)).view(b, c, h, w) + self.pe(v.reshape(b, c, h, w))
        return self.proj(y)


class PSABlock(nn.Module):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True):
        super().__init__()
        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        x = x + self.attn(x) if self.add else self.attn(x)
        return x + self.ffn(x) if self.add else self.ffn(x)


class C3k2(C2f):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, attn=False, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            nn.Sequential(
                Bottleneck(self.c, self.c, shortcut, g),
                PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)),
            )
            if attn
            else C3k(self.c, self.c, 2, shortcut, g)
            if c3k
            else Bottleneck(self.c, self.c, shortcut, g)
            for _ in range(n)
        )


class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5, n=3, shortcut=False):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1, act=False)
        self.cv2 = Conv(c_ * (n + 1), c2, 1, 1)
        self.m = nn.MaxPool2d(k, 1, k // 2)
        self.n = n
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(self.n))
        y = self.cv2(torch.cat(y, 1))
        return y + x if self.add else y


class C2PSA(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        if c1 != c2:
            raise ValueError("C2PSA requires equal input/output channels")
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        self.m = nn.Sequential(*(PSABlock(self.c, 0.5, max(self.c // 64, 1)) for _ in range(n)))

    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        return self.cv2(torch.cat((a, self.m(b)), 1))


class Concat(nn.Module):
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class MaskAttentionFusion(nn.Module):
    """Use a mask feature map to spatially emphasize an image feature map."""

    def __init__(self, channels: int):
        super().__init__()
        self.attention = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, image_feature: torch.Tensor, mask_feature: torch.Tensor) -> torch.Tensor:
        if image_feature.shape != mask_feature.shape:
            raise ValueError(
                "image and mask features must have the same shape, got "
                f"{tuple(image_feature.shape)} and {tuple(mask_feature.shape)}"
            )
        scale = 1.0 + self.attention(mask_feature).sigmoid()
        return image_feature * scale


def make_anchors(feats: list[torch.Tensor], strides: torch.Tensor, offset=0.5):
    points, stride_tensor = [], []
    dtype, device = feats[0].dtype, feats[0].device
    for feat, stride in zip(feats, strides):
        h, w = feat.shape[-2:]
        sy = torch.arange(h, device=device, dtype=dtype) + offset
        sx = torch.arange(w, device=device, dtype=dtype) + offset
        yy, xx = torch.meshgrid(sy, sx, indexing="ij")
        points.append(torch.stack((xx, yy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), float(stride), device=device, dtype=dtype))
    return torch.cat(points), torch.cat(stride_tensor)


def dist2bbox(distance: torch.Tensor, anchor_points: torch.Tensor) -> torch.Tensor:
    lt, rb = distance.chunk(2, -1)
    return torch.cat((anchor_points - lt, anchor_points + rb), -1)


class Detect(nn.Module):
    """Dual one-to-many/one-to-one, DFL-free YOLO26 detection head."""

    def __init__(self, nc=80, reg_max=1, end2end=True, ch=()):
        super().__init__()
        self.nc, self.nl, self.reg_max = nc, len(ch), reg_max
        self.no = nc + reg_max * 4
        self.register_buffer("stride", torch.tensor([8.0, 16.0, 32.0]))
        c2 = max(16, ch[0] // 4, reg_max * 4)
        c3 = max(ch[0], min(nc, 100))
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * reg_max, 1)) for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, nc, 1),
            )
            for x in ch
        )
        self.one2one_cv2 = copy.deepcopy(self.cv2) if end2end else None
        self.one2one_cv3 = copy.deepcopy(self.cv3) if end2end else None
        self.reset_biases()

    def reset_biases(self, image_size=640):
        branches = [(self.cv2, self.cv3)]
        if self.one2one_cv2 is not None:
            branches.append((self.one2one_cv2, self.one2one_cv3))
        for boxes, classes in branches:
            for i, (box, cls) in enumerate(zip(boxes, classes)):
                nn.init.constant_(box[-1].bias, 2.0)
                nn.init.constant_(cls[-1].bias, math.log(5 / self.nc / (image_size / self.stride[i]) ** 2))

    def _branch(self, feats, box_head, cls_head):
        bs = feats[0].shape[0]
        boxes = torch.cat([box_head[i](x).view(bs, 4, -1) for i, x in enumerate(feats)], -1)
        scores = torch.cat([cls_head[i](x).view(bs, self.nc, -1) for i, x in enumerate(feats)], -1)
        return {"boxes": boxes, "scores": scores, "feats": feats}

    def forward(self, feats):
        one2many = self._branch(feats, self.cv2, self.cv3)
        one2one = self._branch([x.detach() for x in feats], self.one2one_cv2, self.one2one_cv3)
        return {"one2many": one2many, "one2one": one2one}

    def decode(self, raw: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        points, strides = make_anchors(raw["feats"], self.stride)
        distances = raw["boxes"].permute(0, 2, 1)
        boxes = dist2bbox(distances, points.unsqueeze(0)) * strides.unsqueeze(0)
        return boxes, raw["scores"].permute(0, 2, 1).sigmoid()

    @torch.no_grad()
    def postprocess(self, raw, conf=0.25, iou=0.7, max_det=300, end2end=True, multi_label=False):
        boxes, scores = self.decode(raw["one2one" if end2end else "one2many"])
        outputs = []
        for b, s in zip(boxes, scores):
            if end2end:
                k = min(max_det, b.shape[0])
                anchors = s.max(-1).values.topk(k).indices
                flat_scores = s[anchors].flatten()
                k = min(max_det, flat_scores.numel())
                confs, flat_idx = flat_scores.topk(k)
                labels = flat_idx % self.nc
                chosen_boxes = b[anchors[flat_idx // self.nc]]
                keep = confs >= conf
                outputs.append(torch.cat((chosen_boxes[keep], confs[keep, None], labels[keep, None].float()), 1))
            else:
                if multi_label:
                    anchor_idx, labels = torch.where(s >= conf)
                    bb, cc, ll = b[anchor_idx], s[anchor_idx, labels], labels
                else:
                    confs, labels = s.max(-1)
                    keep = confs >= conf
                    bb, cc, ll = b[keep], confs[keep], labels[keep]
                if bb.numel():
                    offsets = ll.to(bb) * (bb.max().detach() + 1)
                    idx = nms(bb + offsets[:, None], cc, iou)[:max_det]
                    outputs.append(torch.cat((bb[idx], cc[idx, None], ll[idx, None].float()), 1))
                else:
                    outputs.append(b.new_zeros((0, 6)))
        return outputs


class YOLO26(nn.Module):
    """YOLO26 detector with size in {n, s, m, l, x}."""

    def __init__(self, nc=80, size="n"):
        super().__init__()
        if size not in MODEL_SCALES:
            raise ValueError(f"unknown model size {size!r}; choose from {tuple(MODEL_SCALES)}")
        depth, width, max_channels = MODEL_SCALES[size]
        ch = lambda c: make_divisible(min(c, max_channels) * width, 8)
        rep = lambda n: max(round(n * depth), 1)
        c64, c128, c256, c512, c1024 = (ch(x) for x in (64, 128, 256, 512, 1024))
        large_c3k = size in {"m", "l", "x"}  # Ultralytics parser switches C3k2 internals for larger scales.

        self.nc, self.size = nc, size
        self.model_route = "image"
        self.requires_mask = False
        self.model = nn.ModuleList(
            [
                Conv(3, c64, 3, 2),
                Conv(c64, c128, 3, 2),
                C3k2(c128, c256, rep(2), large_c3k, 0.25),
                Conv(c256, c256, 3, 2),
                C3k2(c256, c512, rep(2), large_c3k, 0.25),
                Conv(c512, c512, 3, 2),
                C3k2(c512, c512, rep(2), True),
                Conv(c512, c1024, 3, 2),
                C3k2(c1024, c1024, rep(2), True),
                SPPF(c1024, c1024, 5, 3, True),
                C2PSA(c1024, c1024, rep(2)),
                nn.Upsample(scale_factor=2, mode="nearest"),
                Concat(1),
                C3k2(c1024 +    c512, c512, rep(2), True),
                nn.Upsample(scale_factor=2, mode="nearest"),
                Concat(1),
                C3k2(c512 + c512, c256, rep(2), True),
                Conv(c256, c256, 3, 2),
                Concat(1),
                C3k2(c256 + c512, c512, rep(2), True),
                Conv(c512, c512, 3, 2),
                Concat(1),
                C3k2(c512 + c1024, c1024, rep(1), True, 0.5, True),
                Detect(nc, 1, True, (c256, c512, c1024)),
            ]
        )
        self._initialize_weights()

    @property
    def head(self) -> Detect:
        return self.model[-1]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps, m.momentum = 1e-3, 0.03
            elif isinstance(m, (nn.ReLU, nn.LeakyReLU, nn.ReLU6, nn.Hardswish)):
                m.inplace = True

    def forward(self, x):
        # Backbone: retain the three feature maps consumed by the neck.
        x = self.model[0](x)
        x = self.model[1](x)
        x = self.model[2](x)
        x = self.model[3](x)
        p3_backbone = self.model[4](x)
        x = self.model[5](p3_backbone)
        p4_backbone = self.model[6](x)
        x = self.model[7](p4_backbone)
        x = self.model[8](x)
        x = self.model[9](x)
        p5_backbone = self.model[10](x)

        # Top-down FPN.
        x = self.model[11](p5_backbone)
        x = self.model[12]([x, p4_backbone])
        p4_fpn = self.model[13](x)
        x = self.model[14](p4_fpn)
        x = self.model[15]([x, p3_backbone])
        p3 = self.model[16](x)

        # Bottom-up PAN.
        x = self.model[17](p3)
        x = self.model[18]([x, p4_fpn])
        p4 = self.model[19](x)
        x = self.model[20](p4)
        x = self.model[21]([x, p5_backbone])
        p5 = self.model[22](x)

        return self.model[23]([p3, p4, p5])

    @torch.no_grad()
    def predict(self, images, conf=0.25, iou=0.7, max_det=300, end2end=True):
        training = self.training
        self.eval()
        raw = self(images)
        result = self.head.postprocess(raw, conf, iou, max_det, end2end)
        self.train(training)
        return result

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def load_compact(self, path: str | Path, strict=True):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = ckpt.get("ema") or ckpt.get("model") or ckpt
        if isinstance(state, nn.Module):
            state = state.state_dict()
        if not strict:
            current = self.state_dict()
            state = {k: v for k, v in state.items() if k in current and current[k].shape == v.shape}
        return self.load_state_dict(state, strict=strict)


class YOLO26withMaskGuider(YOLO26):
    """YOLO26 with a mask backbone that guides P3/P4/P5 image features.

    The mask branch mirrors image-backbone layers 0 through 8. Its outputs after
    layers 4, 6 and 8 generate single-channel spatial attention maps. Each map is
    applied residually as ``image_feature * (1 + sigmoid(attention))``.
    """

    def __init__(self, nc=80, size="n", mask_channels=1):
        super().__init__(nc=nc, size=size)
        if mask_channels < 1:
            raise ValueError(f"mask_channels must be positive, got {mask_channels}")

        image_stem = self.model[0]
        stem_channels = image_stem.conv.out_channels
        self.mask_channels = mask_channels
        self.model_route = "mask_guider"
        self.requires_mask = True
        self.mask_backbone = nn.ModuleList(
            [
                Conv(mask_channels, stem_channels, 3, 2),
                *[copy.deepcopy(module) for module in self.model[1:9]],
            ]
        )

        p3_channels = self.model[4].cv2.conv.out_channels
        p4_channels = self.model[6].cv2.conv.out_channels
        p5_channels = self.model[8].cv2.conv.out_channels
        self.mask_fusion = nn.ModuleList(
            [
                MaskAttentionFusion(p3_channels),
                MaskAttentionFusion(p4_channels),
                MaskAttentionFusion(p5_channels),
            ]
        )
        self._initialize_weights()

    def _mask_features(self, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = mask
        p3 = p4 = None
        for index, module in enumerate(self.mask_backbone):
            x = module(x)
            if index == 4:
                p3 = x
            elif index == 6:
                p4 = x
        return p3, p4, x

    def forward(self, image: torch.Tensor, mask: torch.Tensor):
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        if image.ndim != 4 or mask.ndim != 4:
            raise ValueError("image and mask must be BCHW tensors")
        if mask.shape[1] != self.mask_channels:
            raise ValueError(f"expected a {self.mask_channels}-channel mask, got {mask.shape[1]} channels")
        if image.shape[0] != mask.shape[0] or image.shape[-2:] != mask.shape[-2:]:
            raise ValueError(
                "image and mask must have the same batch and spatial dimensions, got "
                f"{tuple(image.shape)} and {tuple(mask.shape)}"
            )

        mask_p3, mask_p4, mask_p5 = self._mask_features(mask)

        # Image backbone with mask-guided fusion at P3, P4 and P5.
        x = self.model[0](image)
        x = self.model[1](x)
        x = self.model[2](x)
        x = self.model[3](x)
        p3_backbone = self.mask_fusion[0](self.model[4](x), mask_p3)
        x = self.model[5](p3_backbone)
        p4_backbone = self.mask_fusion[1](self.model[6](x), mask_p4)
        x = self.model[7](p4_backbone)
        x = self.mask_fusion[2](self.model[8](x), mask_p5)
        x = self.model[9](x)
        p5_backbone = self.model[10](x)

        # Standard YOLO26 FPN/PAN neck.
        x = self.model[11](p5_backbone)
        x = self.model[12]([x, p4_backbone])
        p4_fpn = self.model[13](x)
        x = self.model[14](p4_fpn)
        x = self.model[15]([x, p3_backbone])
        p3 = self.model[16](x)

        x = self.model[17](p3)
        x = self.model[18]([x, p4_fpn])
        p4 = self.model[19](x)
        x = self.model[20](p4)
        x = self.model[21]([x, p5_backbone])
        p5 = self.model[22](x)
        return self.model[23]([p3, p4, p5])

    @torch.no_grad()
    def predict(self, images, masks, conf=0.25, iou=0.7, max_det=300, end2end=True):
        training = self.training
        self.eval()
        raw = self(images, masks)
        result = self.head.postprocess(raw, conf, iou, max_det, end2end)
        self.train(training)
        return result


def build_model(
    size="n", nc=80, weights: str | None = None, model_route="image", mask_channels=1
) -> YOLO26:
    routes = {
        "image": YOLO26,
        "mask_guider": YOLO26withMaskGuider,
    }
    if model_route not in routes:
        raise ValueError(f"unknown model route {model_route!r}; choose from {tuple(routes)}")
    kwargs = {"mask_channels": mask_channels} if model_route == "mask_guider" else {}
    model = routes[model_route](nc=nc, size=size, **kwargs)
    if weights:
        model.load_compact(weights, strict=False)
    return model
