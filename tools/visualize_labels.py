from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

ROOT = Path("/mnt/workspace/ty/xray/datasets/png")
IMAGE_DIR = ROOT / "images"
LABEL_DIR = ROOT / "labels"
VIS_DIR = ROOT / "vis"

CLASS_NAMES = {
    0: "Aortic enlargement",
    1: "Atelectasis",
    2: "Calcification",
    3: "Cardiomegaly",
    4: "Consolidation",
    5: "ILD",
    6: "Infiltration",
    7: "Lung Opacity",
    8: "Nodule/Mass",
    9: "Other lesion",
    10: "Pleural effusion",
    11: "Pleural thickening",
    12: "Pneumothorax",
    13: "Pulmonary fibrosis",
    14: "No finding",
}

VIS_DIR.mkdir(parents=True, exist_ok=True)

def get_font(size=16):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ]:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()

def draw_yolo(image_path, label_path, output_path):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = get_font(16)
    w, h = img.size

    if not label_path.exists():
        img.save(output_path)
        return

    lines = label_path.read_text().strip().splitlines()
    for line in lines:
        if not line.strip():
            continue

        parts = line.split()
        if len(parts) < 5:
            continue

        cls_id = int(float(parts[0]))
        xc, yc, bw, bh = map(float, parts[1:5])

        x1 = (xc - bw / 2) * w
        y1 = (yc - bh / 2) * h
        x2 = (xc + bw / 2) * w
        y2 = (yc + bh / 2) * h

        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))

        name = CLASS_NAMES.get(cls_id, str(cls_id))
        text = f"{cls_id}: {name}"

        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)

        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        ty = max(0, y1 - th - 6)

        draw.rectangle([x1, ty, x1 + tw + 8, ty + th + 6], fill=(255, 0, 0))
        draw.text((x1 + 4, ty + 2), text, fill=(255, 255, 255), font=font)

    img.save(output_path, quality=95)

def main():
    exts = {".png", ".jpg", ".jpeg"}
    images = [p for p in IMAGE_DIR.iterdir() if p.suffix.lower() in exts]

    print(f"Found {len(images)} images")

    for image_path in tqdm(images):
        label_path = LABEL_DIR / f"{image_path.stem}.txt"
        output_path = VIS_DIR / f"{image_path.stem}.jpg"
        draw_yolo(image_path, label_path, output_path)

    print(f"Saved to: {VIS_DIR}")

if __name__ == "__main__":
    main()