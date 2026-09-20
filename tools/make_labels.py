from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from ensemble_boxes import weighted_boxes_fusion
from tqdm import tqdm

CSV_PATH = "/mnt/workspace/ty/xray/datasets/dicom/train.csv"
DICOM_DIR = Path("/mnt/workspace/ty/xray/datasets/dicom/train")
LABEL_DIR = Path("/mnt/workspace/ty/xray/datasets/png/labels")
OUTPUT_CSV = "/mnt/workspace/ty/xray/datasets/dicom/train_with_size.csv"

IMG_SIZE = 640
NO_FINDING_ID = 14
IOU_THR = 0.5

def build_dicom_index(root):
    return {p.stem: p for p in root.rglob("*") if p.is_file()}

def get_size(path):
    ds = pydicom.dcmread(path, stop_before_pixels=True)
    return int(ds.Columns), int(ds.Rows)

def letterbox_box(box, w, h, size=640):
    x1, y1, x2, y2 = box
    scale = min(size / w, size / h)
    nw, nh = round(w * scale), round(h * scale)
    sx, sy = nw / w, nh / h
    px, py = (size - nw) // 2, (size - nh) // 2
    return x1 * sx + px, y1 * sy + py, x2 * sx + px, y2 * sy + py

def xyxy_to_yolo(box, size=640):
    x1, y1, x2, y2 = box
    xc = ((x1 + x2) / 2) / size
    yc = ((y1 + y2) / 2) / size
    bw = (x2 - x1) / size
    bh = (y2 - y1) / size
    return xc, yc, bw, bh

def fuse_class(df, w, h):
    boxes_list, scores_list, labels_list = [], [], []
    for _, rdf in df.groupby("rad_id"):
        boxes = []
        for _, r in rdf.iterrows():
            box = [r.x_min / w, r.y_min / h, r.x_max / w, r.y_max / h]
            box = np.clip(box, 0, 1).tolist()
            if box[2] > box[0] and box[3] > box[1]:
                boxes.append(box)
        if boxes:
            boxes_list.append(boxes)
            scores_list.append([1.0] * len(boxes))
            labels_list.append([0] * len(boxes))
    if not boxes_list:
        return []
    boxes, scores, _ = weighted_boxes_fusion(
        boxes_list, scores_list, labels_list,
        iou_thr=IOU_THR, skip_box_thr=0.0
    )
    return [(b[0] * w, b[1] * h, b[2] * w, b[3] * h) for b in boxes]

def main():
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV_PATH)
    dicom_index = build_dicom_index(DICOM_DIR)

    sizes = {}
    failed = []

    for image_id in tqdm(df.image_id.unique(), desc="Read DICOM size"):
        path = dicom_index.get(image_id)
        if path is None:
            failed.append((image_id, "DICOM not found"))
            continue
        try:
            sizes[image_id] = get_size(path)
        except Exception as e:
            failed.append((image_id, str(e)))

    df["width"] = df["image_id"].map(lambda x: sizes.get(x, (None, None))[0])
    df["height"] = df["image_id"].map(lambda x: sizes.get(x, (None, None))[1])
    df.to_csv(OUTPUT_CSV, index=False)

    for image_id, g in tqdm(df.groupby("image_id"), total=df.image_id.nunique(), desc="Create YOLO labels"):
        if image_id not in sizes:
            continue

        w, h = sizes[image_id]
        det = g[(g.class_id != NO_FINDING_ID)].dropna(subset=["x_min", "y_min", "x_max", "y_max"])
        lines = []

        for class_id, cg in det.groupby("class_id"):
            for box in fuse_class(cg, w, h):
                box = letterbox_box(box, w, h, IMG_SIZE)
                xc, yc, bw, bh = xyxy_to_yolo(box, IMG_SIZE)

                xc, yc = np.clip([xc, yc], 0, 1)
                bw, bh = np.clip([bw, bh], 0, 1)

                if bw > 0 and bh > 0:
                    lines.append(f"{int(class_id)} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        with open(LABEL_DIR / f"{image_id}.txt", "w") as f:
            f.write("\n".join(lines))

    if failed:
        pd.DataFrame(failed, columns=["image_id", "error"]).to_csv("failed.csv", index=False)

    print(f"Saved CSV: {OUTPUT_CSV}")
    print(f"Saved labels: {LABEL_DIR}")
    print(f"Failed: {len(failed)}")

if __name__ == "__main__":
    main()