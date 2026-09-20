import os
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import torch
import torchxrayvision as xrv

# =========================
# Config
# =========================
device = 'cuda:1'
IMG_DIR = Path("datasets/png/images")
SEG_DIR = "datasets/png/seg"
SEG_VIS_DIR = "datasets/png/seg_vis"
Path(SEG_DIR).mkdir(exist_ok=True)
Path(SEG_VIS_DIR).mkdir(exist_ok=True)

THRESH = 0.5
ALPHA = 0.35   # độ trong suốt lớp màu

# 14 màu BGR khác nhau
COLORS = [
    (255,   0,   0),   # Left Clavicle
    (  0, 255,   0),   # Right Clavicle
    (  0,   0, 255),   # Left Scapula
    (255, 255,   0),   # Right Scapula
    (255,   0, 255),   # Left Lung
    (  0, 255, 255),   # Right Lung
    (128,   0,   0),   # Left Hilus Pulmonis
    (  0, 128,   0),   # Right Hilus Pulmonis
    (  0,   0, 128),   # Heart
    (128, 128,   0),   # Aorta
    (128,   0, 128),   # Facies Diaphragmatica
    (  0, 128, 128),   # Mediastinum
    (200, 100,   0),   # Weasand
    (100,   0, 200),   # Spine
]


def ensure_probs(x):
    """
    Nếu output là logits thì sigmoid.
    Nếu đã là prob [0,1] thì giữ nguyên.
    """
    if x.min() < 0 or x.max() > 1:
        x = torch.sigmoid(x)
    return x


def overlay_mask(image_bgr, mask, color, alpha=0.35):
    """
    image_bgr: HxWx3
    mask: HxW bool / uint8
    color: BGR tuple
    """
    out = image_bgr.copy()
    mask = mask.astype(bool)

    color_arr = np.array(color, dtype=np.float32)
    out_float = out.astype(np.float32)

    out_float[mask] = out_float[mask] * (1 - alpha) + color_arr * alpha
    out = out_float.astype(np.uint8)

    # vẽ contour để rõ biên
    mask_u8 = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, 1)

    return out


def add_legend(image_bgr, class_names, colors):
    h, w = image_bgr.shape[:2]
    legend_w = 360
    canvas = np.ones((h, w + legend_w, 3), dtype=np.uint8) * 255
    canvas[:, :w] = image_bgr

    x0 = w + 20
    y = 30
    cv2.putText(canvas, "Classes", (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
    y += 25

    for name, color in zip(class_names, colors):
        cv2.rectangle(canvas, (x0, y - 12), (x0 + 20, y + 8), color, -1)
        cv2.rectangle(canvas, (x0, y - 12), (x0 + 20, y + 8), (0,0,0), 1)
        cv2.putText(canvas, name, (x0 + 30, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 1, cv2.LINE_AA)
        y += 28

    return canvas


# =========================
# Load model
# =========================
model = xrv.baseline_models.chestx_det.PSPNet()
model.eval()
model.to(device)

# =========================
# Read original image
# =========================
for img_path in IMG_DIR.glob('*.png'):
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise RuntimeError(f"Cannot read image: {img_path}")

    orig_h, orig_w = img_gray.shape[:2]

    # ảnh để visualize
    vis_img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    # =========================
    # Preprocess for model
    # =========================
    img = xrv.datasets.normalize(img_gray, 255)   # -> float
    img = img[None, ...]                          # [1, H, W]

    transform = xrv.datasets.XRayResizer(512)
    img = transform(img)                          # [1, 512, 512]

    img = torch.from_numpy(img).float().unsqueeze(0).to(device)   # [1,1,512,512]

    # =========================
    # Inference
    # =========================
    with torch.no_grad():
        output = model(img)                       # [1,14,512,512]

    output = output[0].cpu()                      # [14,512,512]
    output = ensure_probs(output)

    class_names = model.targets
    print("Classes:", class_names)

    # =========================
    # Resize masks back to original size + overlay
    # =========================
    overlay = vis_img.copy()
    masks = []
    for i, class_name in enumerate(class_names):
        if i not in [4, 5, 8, 9]:
            continue
        prob_map = output[i].numpy()  # [512,512]

        # resize về size ảnh gốc
        prob_map = cv2.resize(prob_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        # threshold để ra binary mask
        mask = prob_map > THRESH

        # nếu muốn bỏ class quá nhỏ thì có thể check mask.sum()
        if mask.sum() == 0:
            continue

        overlay = overlay_mask(overlay, mask, COLORS[i], alpha=ALPHA)
        masks.append(mask)

        print(f"{i:02d} | {class_name:25s} | area={int(mask.sum())}")

    # =========================
    # Add legend and save
    # =========================
    result = add_legend(overlay, class_names, COLORS)
    cv2.imwrite(f'{SEG_VIS_DIR}/{os.path.basename(img_path)}', result)
    np.save(f'{SEG_DIR}/{os.path.basename(img_path)[:-4]}.npy', np.stack(masks, 0))
