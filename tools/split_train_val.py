import random
import shutil
from pathlib import Path


# =========================
# Config - sua truc tiep o day
# =========================
IMAGE_DIR = Path("datasets/png/images")
LABEL_DIR = Path("datasets/png/labels")
SAVE_DIR = Path("datasets/yolo")

# Co the nhap 80 / 20 hoac 0.8 / 0.2
TRAIN_PERCENT = 80
VAL_PERCENT = 20

# So luong anh background label rong muon them vao dataset
NUM_BACKGROUND = 0

SEED = 42
COPY_MODE = "copy"   # "copy" hoac "symlink"
OVERWRITE = True     # True: xoa SAVE_DIR cu roi tao lai
CLASS_NAMES = [
    "Aortic enlargement",
    "Atelectasis",
    "Calcification",
    "Cardiomegaly",
    "Consolidation",
    "ILD",
    "Infiltration",
    "Lung Opacity",
    "Nodule/Mass",
    "Other lesion",
    "Pleural effusion",
    "Pleural thickening",
    "Pneumothorax",
    "Pulmonary fibrosis",
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def normalize_percent(value):
    if value > 1:
        value = value / 100
    return value


def is_background(label_path):
    return (not label_path.exists()) or label_path.read_text().strip() == ""


def collect_samples(image_dir, label_dir):
    positives = []
    backgrounds = []

    for image_path in sorted(image_dir.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTS:
            continue

        label_path = label_dir / f"{image_path.stem}.txt"
        sample = (image_path, label_path)

        if is_background(label_path):
            backgrounds.append(sample)
        else:
            positives.append(sample)

    return positives, backgrounds


def split_samples(samples, train_ratio, rng):
    samples = list(samples)
    rng.shuffle(samples)

    n_train = round(len(samples) * train_ratio)
    n_train = min(max(n_train, 0), len(samples))

    return samples[:n_train], samples[n_train:]


def prepare_output(save_dir, overwrite):
    if save_dir.exists() and overwrite:
        shutil.rmtree(save_dir)

    for split in ("train", "val"):
        (save_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (save_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def put_file(src, dst, mode):
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def export_samples(samples, split, save_dir, copy_mode):
    image_out_dir = save_dir / "images" / split
    label_out_dir = save_dir / "labels" / split

    for image_path, label_path in samples:
        put_file(image_path, image_out_dir / image_path.name, copy_mode)

        out_label = label_out_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            put_file(label_path, out_label, copy_mode)
        else:
            out_label.write_text("")


def write_data_yaml(save_dir, class_names):
    lines = [
        f"path: {save_dir.resolve()}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(class_names)}",
        "names:",
    ]
    lines.extend([f"  {i}: {name}" for i, name in enumerate(class_names)])
    (save_dir / "data.yaml").write_text("\n".join(lines) + "\n")


def main():
    train_ratio = normalize_percent(TRAIN_PERCENT)
    val_ratio = normalize_percent(VAL_PERCENT) if VAL_PERCENT is not None else 1 - train_ratio

    if train_ratio < 0 or val_ratio < 0:
        raise ValueError("train/val percent phai >= 0.")
    if abs((train_ratio + val_ratio) - 1) > 1e-6:
        raise ValueError("Tong train_percent va val_percent phai bang 100% hoac 1.0.")
    if NUM_BACKGROUND < 0:
        raise ValueError("num_background phai >= 0.")
    if COPY_MODE not in {"copy", "symlink"}:
        raise ValueError('COPY_MODE phai la "copy" hoac "symlink".')

    rng = random.Random(SEED)
    positives, backgrounds = collect_samples(IMAGE_DIR, LABEL_DIR)

    if NUM_BACKGROUND > len(backgrounds):
        print(
            f"Warning: chi co {len(backgrounds)} background, "
            f"nhung yeu cau {NUM_BACKGROUND}. Se lay tat ca."
        )

    rng.shuffle(backgrounds)
    selected_backgrounds = backgrounds[:NUM_BACKGROUND]

    train_pos, val_pos = split_samples(positives, train_ratio, rng)
    train_bg, val_bg = split_samples(selected_backgrounds, train_ratio, rng)

    train_samples = train_pos + train_bg
    val_samples = val_pos + val_bg
    rng.shuffle(train_samples)
    rng.shuffle(val_samples)

    prepare_output(SAVE_DIR, OVERWRITE)
    export_samples(train_samples, "train", SAVE_DIR, COPY_MODE)
    export_samples(val_samples, "val", SAVE_DIR, COPY_MODE)
    write_data_yaml(SAVE_DIR, CLASS_NAMES)

    print(f"Images co label: {len(positives)}")
    print(f"Background available: {len(backgrounds)}")
    print(f"Background selected: {len(selected_backgrounds)}")
    print(f"Train: {len(train_samples)} ({len(train_pos)} label, {len(train_bg)} background)")
    print(f"Val: {len(val_samples)} ({len(val_pos)} label, {len(val_bg)} background)")
    print(f"Saved to: {SAVE_DIR}")
    print(f"Data yaml: {SAVE_DIR / 'data.yaml'}")


if __name__ == "__main__":
    main()
