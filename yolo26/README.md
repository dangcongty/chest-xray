# YOLO26 Compact — detection-only

Một bản tách gọn pipeline **object detection** của Ultralytics YOLO26 thành các file dễ đọc và dễ sửa. Dự án không kéo theo framework đa-task của Ultralytics, nhưng vẫn giữ đúng các phần cốt lõi của YOLO26 detection:

- model scale `n / s / m / l / x`;
- backbone + PAN/FPN + `C3k2`, `C2PSA`, `SPPF`;
- dual head `one2many` và `one2one`;
- DFL-free box regression (`reg_max=1`);
- Task-Aligned Assignment + STAL cho vật thể nhỏ;
- Progressive Loss chuyển trọng số từ nhánh one-to-many sang one-to-one;
- inference NMS-free hoặc one-to-many + NMS;
- optimizer AdamW, SGD và MuSGD;
- train, resume, EMA, validation mAP50/mAP50-95, predict và chuyển official checkpoint.

## Cấu trúc

| File | Trách nhiệm |
| --- | --- |
| `model.py` | Toàn bộ kiến trúc YOLO26 detection và decode/postprocess |
| `dataloader.py` | Đọc ảnh + nhãn YOLO txt, letterbox, HSV, flip, mosaic-4 |
| `losses.py` | TAL/STAL assigner, CIoU, BCE, L1 và Progressive Loss |
| `optimizer.py` | MuSGD và optimizer builder |
| `metrics.py` | Matching, precision, recall, AP và mAP |
| `train.py` | Vòng lặp train, AMP, EMA, resume, early stopping, checkpoint |
| `val.py` | Validation độc lập |
| `predict.py` | Inference ảnh và vẽ kết quả |
| `convert_ultralytics.py` | Chuyển `yolo26*.pt` chính thức sang checkpoint compact |
| `config.py`, `utils.py` | Cấu hình và helper dùng chung |

## Cài đặt

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Dataset

Giữ format detection quen thuộc của Ultralytics:

```text
datasets/my_data/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/   # mỗi ảnh có file .txt tương ứng
    └── val/
```

Mỗi dòng label là `class x_center y_center width height`, tọa độ chuẩn hóa về `[0, 1]`.

```yaml
# data.yaml
path: /absolute/path/to/datasets/my_data
train: images/train
val: images/val
names: [person, car, motorbike]
```

## Train

Từ scratch:

```bash
python train.py --data data.yaml --size n --epochs 100 --batch-size 16 --device 0
```

Dùng file cấu hình:

```bash
python train.py --config configs/train_n.yaml --data data.yaml
```

Resume đầy đủ optimizer/epoch:

```bash
python train.py --data data.yaml --resume runs/train/exp/weights/last.pt
```

### Dùng pretrained chính thức

Chuyển một lần bằng package Ultralytics:

```bash
pip install ultralytics
python convert_ultralytics.py --weights yolo26n.pt
python train.py --data data.yaml --size n --weights yolo26n.compact.pt
```

Khi số class của dataset khác COCO, loader tự bỏ các tensor classification head không cùng shape và vẫn nạp toàn bộ backbone/neck/phần head tương thích.

## Validate

Mặc định validation dùng nhánh one-to-many + NMS, giống lựa chọn thiên về accuracy:

```bash
python val.py --weights runs/train/exp/weights/best.pt --data data.yaml
```

Đánh giá nhánh one-to-one NMS-free:

```bash
python val.py --weights runs/train/exp/weights/best.pt --data data.yaml --end2end
```

## Predict

NMS-free:

```bash
python predict.py --weights runs/train/exp/weights/best.pt --source path/to/images
```

One-to-many + NMS:

```bash
python predict.py --weights runs/train/exp/weights/best.pt --source path/to/images --nms
```

## Điều gì đã được lược bỏ?

Bản compact chủ động không chứa semantic segmentation, depth, classification, pose, OBB, tracking, exporter, HUB/cloud, tuning, callbacks, plotting phức tạp và multi-scale training. Trainer hiện hỗ trợ DDP nhiều GPU. Mosaic ở đây là bản 4 ô cố định, không phải toàn bộ chuỗi augmentation của Ultralytics. Vì vậy:

- kiến trúc detection, raw output và loss được giữ tương thích;
- official detection weights chuyển được;
- kết quả train mới **không được cam kết bit-identical** với trainer Ultralytics đầy đủ;
- nếu cần reproduce đúng benchmark COCO chính thức, vẫn nên dùng upstream Ultralytics và đúng recipe/checkpoint revision.

## Kiểm tra

```bash
python -m unittest discover -s tests -v
```

## Nguồn và giấy phép

Code được viết lại từ Ultralytics repository tại commit `6900c83b16eebee55c7b9de23b9ef447e6ff11e7` (2026-09-17), tập trung duy nhất vào detect. Kiến trúc tham chiếu `ultralytics/cfg/models/26/yolo26.yaml`; head ở `ultralytics/nn/modules/head.py`; loss/assigner ở `ultralytics/utils/loss.py` và `ultralytics/utils/tal.py`.

Dự án phái sinh được phát hành theo **GNU Affero General Public License v3.0 only (AGPL-3.0-only)**. Xem `LICENSE` và `NOTICE.md`.
