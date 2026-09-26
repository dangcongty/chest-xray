# Coarse-guided YOLO26 for Chest X-ray Detection

Repository này triển khai YOLO26 compact cho phát hiện bất thường trên X-quang
ngực, đồng thời xử lý một dạng không nhất quán phổ biến trong annotation: cùng
một vùng bệnh, một bác sĩ khoanh vùng lớn (đôi khi gần toàn bộ phổi), trong khi
các bác sĩ khác khoanh nhiều tổn thương nhỏ bên trong.

Thay vì coi cả hai loại box là các object ngang hàng, phương pháp hiện tại tách
chúng thành hai nhiệm vụ:

1. **Coarse localization:** box lớn cho biết vùng chắc chắn có bệnh và dùng để
   giám sát một guide head.
2. **Fine detection:** box nhỏ dùng để học class và tọa độ tổn thương chính xác.

## Pipeline

```text
YOLO labels
    |
    +-- box lớn chứa box nhỏ cùng class --> class-aware guide mask
    |
    +-- các box còn lại -----------------> fine detection targets

Image --> YOLO26 backbone --> P3/P4/P5 --> coarse guide heads
                               |              |
                               |              +--> guide logits + guide loss
                               |              |
                               +<-- residual spatial attention
                               |
                               +--> FPN/PAN --> detection heads --> fine boxes
```

Guide được model tự dự đoán từ ảnh. Ground-truth guide chỉ dùng để tính loss khi
train, vì vậy inference không cần box lớn hoặc mask do bác sĩ cung cấp.

## Tách coarse box và fine box

Với mỗi ảnh, một box được chuyển thành coarse guide khi thỏa cả hai nhóm điều
kiện sau:

- là box lớn: diện tích chuẩn hóa `>= 0.10`, hoặc chiều rộng `>= 0.35`, hoặc
  chiều cao `>= 0.50`;
- chứa ít nhất một box nhỏ cùng class với:
  - `IoBB(small, large) >= 0.8`;
  - `area(large) / area(small) >= 4.0`.

Trong đó:

```text
IoBB(small, large) = intersection(small, large) / area(small)
```

IoBB được dùng thay IoU vì IoU sẽ rất thấp khi box lớn bao phủ đúng một box nhỏ.
Các coarse box được rasterize thành mask `[num_classes, H, W]`; chúng không còn
tham gia detection loss. Augmentation hình học như horizontal flip và mosaic
được áp dụng đồng bộ cho ảnh, fine box và guide mask.

## Coarse guider

Route `coarse_guider` thêm một convolution `1x1` class-aware tại mỗi mức
P3/P4/P5. Guide probability được gộp theo chiều class thành spatial attention và
fuse theo residual gating:

```text
guided_feature = feature * (1 + max(sigmoid(guide_logits), dim=class))
```

Residual gating giữ nguyên đường truyền feature gốc, nên guide dự đoán chưa tốt
không thể xóa hoàn toàn tín hiệu của tổn thương nhỏ.

Model vẫn hỗ trợ ba route độc lập:

| Route | Mô tả |
| --- | --- |
| `image` | YOLO26 ảnh gốc, không có guider |
| `mask_guider` | Nhận anatomical mask từ bên ngoài qua mask backbone |
| `coarse_guider` | Tự dự đoán guide từ ảnh và học bằng coarse boxes |
| `stn` | Học affine STN trên ảnh trước YOLO26; box nhãn được biến đổi đồng bộ trong loss |

Route `stn` dùng toàn bộ box như `image`. STN bắt đầu ở phép biến đổi đồng nhất,
học dịch chuyển/co giãn/xiên nhẹ cùng YOLO26, và đưa box dự đoán trở về tọa độ
ảnh gốc trước khi tính metric hoặc xuất kết quả. Chọn bằng
`model_route="stn"` trong `train(...)`; không cần sửa dataloader hay nhãn.

## Loss

Tổng loss của route `coarse_guider` là:

```text
L = L_detection + lambda_guide * L_guide
L_guide = BCE_with_logits + Dice
```

`L_detection` giữ nguyên progressive dual-head loss của YOLO26, gồm CIoU box
loss, classification BCE và normalized L1. Guide loss được tính tại cả ba mức
P3/P4/P5; BCE dùng positive weighting có giới hạn để giảm mất cân bằng foreground
và background.

Mặc định `lambda_guide = 0.5`.

## Dataset

Dataset sử dụng YOLO format thông thường:

```text
datasets/yolo/
├── data.yaml
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

Mỗi dòng label:

```text
class_id x_center y_center width height
```

Các tọa độ được chuẩn hóa về `[0, 1]`. Việc tách coarse/fine diễn ra động trong
dataloader, không sửa file nhãn gốc.

## Train

Cấu hình Python tối thiểu:

```python
from train import train

train(
    data="datasets/yolo/data.yaml",
    size="m",
    weights="yolo26m.pt",
    model_route="coarse_guider",
    guide_loss_weight=0.5,
    guide_iobb=0.8,
    guide_min_area_ratio=4.0,
    guide_min_area=0.10,
    guide_min_width=0.35,
    guide_min_height=0.50,
    use_multiclass=False,
    image_size=640,
    batch_size=16,
    epochs=500,
    device=0,
    name="coarse-guider-yolo-attempt-1",
)
```

Route này hiện chỉ hỗ trợ nhãn YOLO single-class. `image` và `mask_guider` giữ
nguyên flow cũ. Có thể truyền `device=[0, 1]` để chạy DDP, nhưng cần đủ RAM hệ
thống cho nhiều process, worker, EMA và checkpoint.

Resume:

```python
train(
    "runs/train/coarse-guider-yolo-attempt-1/args.yaml",
    resume="runs/train/coarse-guider-yolo-attempt-1/weights/last.pt",
    device=0,
)
```

## Validation và cách đọc metric

Các metric `precision`, `recall`, `mAP50` và `mAP50-95` hiện được tính cho
**fine detection boxes**. `guide` trên progress bar là guide loss của batch train
gần nhất, không phải mAP của coarse head.

Validation coarse riêng (pixel IoU, Dice, guide recall và tỷ lệ guide bao phủ
fine boxes) chưa được triển khai. Vì mục tiêu của guide là không bỏ sót vùng chứa
tổn thương, `guide recall/coverage` sẽ quan trọng hơn việc biên mask khớp tuyệt
đối. Không nên diễn giải fine-box mAP hiện tại như chất lượng của coarse guider.

## Cấu trúc chính

| File | Trách nhiệm |
| --- | --- |
| `train.py` | Entry point train một hoặc nhiều GPU |
| `yolo26/model.py` | YOLO26, external mask guider và learned coarse guider |
| `yolo26/dataloader.py` | Tách coarse/fine targets và augmentation đồng bộ |
| `yolo26/losses.py` | Detection loss và BCE + Dice guide loss |
| `yolo26/config.py` | Route và threshold cấu hình |
| `yolo26/val.py` | Validation fine detection |
| `yolo26/tests/test_smoke.py` | Smoke tests cho model, loss và data routes |

## Kiểm tra

```bash
source lib/bin/activate
PYTHONPATH=yolo26 python -m unittest discover -s yolo26/tests -v
```

## Giấy phép

Mã nguồn sử dụng giấy phép `AGPL-3.0-only`. Xem `yolo26/LICENSE` và
`yolo26/NOTICE.md`.
