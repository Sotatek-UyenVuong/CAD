# CAD Drawing YOLO Training Pipeline

Pipeline train YOLO để detect 4 loại element trên bản vẽ CAD kỹ thuật.

## Classes

| ID | Tên | Mô tả |
|----|-----|-------|
| 0 | `text` | Văn bản, chú thích |
| 1 | `table` | Bảng biểu |
| 2 | `title_block` | Khung tên (title block) |
| 3 | `diagram` | Sơ đồ, hình vẽ kỹ thuật |

## Dữ liệu nguồn

| Folder | Loại bản vẽ | Tổng ảnh | Có nhãn | Background |
|--------|-------------|----------|---------|------------|
| `rendered_竣工図（新綱島スクエア　建築意匠図）_dpi300` | Kiến trúc | 361 | 154 | 207 |
| `rendered_竣工図（新綱島スクエア　電気設備図）_dpi300` | Điện | 411 | 411 | 0 |
| `rendered_竣工図（新綱島スクエア　構造図）_dpi300` | Kết cấu | 141 | 141 | 0 |
| `rendered_竣工図（新綱島スクエア　機械設備図）_dpi300` | Cơ điện lạnh | 275 | 107 | 168 |
| **Tổng** | | **1.188** | **813** | **375** |

> **Background samples**: ảnh không có nhãn được đưa vào training với label file rỗng.
> YOLO dùng chúng để học cách *không* detect trên trang trắng/trang không có object,
> giúp giảm false positive đáng kể.

## Cài đặt

```bash
# Tạo virtual environment
python -m venv .venv
source .venv/bin/activate

# Cài đặt dependencies
pip install ultralytics opencv-python-headless pyyaml
```

## Luồng sử dụng

### Bước 1 – Chuẩn bị dataset

```bash
# Mặc định: split 80% train / 10% val / 10% test
python tools/prepare_dataset.py

# Tùy chỉnh tỉ lệ split
python tools/prepare_dataset.py --train-ratio 0.85 --val-ratio 0.10

# Xem tất cả tùy chọn
python tools/prepare_dataset.py --help
```

Output sẽ tạo ra:
```
dataset/
├── images/
│   ├── train/   (948 ảnh = 648 labeled + 300 background)
│   ├── val/     (118 ảnh)
│   └── test/    (122 ảnh)
├── labels/
│   ├── train/   (948 .txt – file rỗng cho background samples)
│   ├── val/
│   └── test/
└── cad_dataset.yaml
```

> Dùng `--labeled-only` để chỉ lấy ảnh có nhãn (813 ảnh, bỏ qua background).

### Bước 2 – Train model

```bash
# Training cơ bản (YOLOv8m, 100 epoch, 1280px) – khuyến nghị
python tools/train.py

# Training nhanh để kiểm tra (YOLOv8n, 10 epoch)
python tools/train.py --model yolov8n.pt --epochs 10 --imgsz 640

# Training đầy đủ với augmentation mạnh (cho dataset nhỏ)
python tools/train.py \
  --model yolov8m.pt \
  --epochs 200 \
  --imgsz 1280 \
  --batch 8 \
  --lr0 0.005 \
  --augment

# Training YOLO11 (kiến trúc mới nhất)
python tools/train.py --model yolo11m.pt --epochs 150

# Resume khi training bị gián đoạn
python tools/train.py --resume runs/detect/cad_yolo/weights/last.pt
```

Kết quả lưu tại: `runs/detect/cad_yolo/`

### Bước 3 – Đánh giá model

```bash
# Evaluate trên test set
python tools/evaluate.py --weights runs/detect/cad_yolo/weights/best.pt

# Chạy inference trên một ảnh cụ thể
python tools/evaluate.py \
  --weights runs/detect/cad_yolo/weights/best.pt \
  --predict dataset/images/test/folder0__page_5.png

# Chạy inference trên toàn bộ test set (để xem kết quả visual)
python tools/evaluate.py \
  --weights runs/detect/cad_yolo/weights/best.pt \
  --predict dataset/images/test/ \
  --conf 0.3

# Xem tất cả tùy chọn
python tools/evaluate.py --help
```

## Lựa chọn model

| Model | Tham số | Tốc độ | Độ chính xác | Khuyến nghị |
|-------|---------|--------|--------------|-------------|
| yolov8n.pt | 3.2M | ★★★★★ | ★★ | Debug/test nhanh |
| yolov8s.pt | 11.2M | ★★★★ | ★★★ | Edge deployment |
| **yolov8m.pt** | 25.9M | ★★★ | ★★★★ | **Cân bằng tốt nhất** |
| yolov8l.pt | 43.7M | ★★ | ★★★★★ | Server inference |
| yolo11m.pt | 20.1M | ★★★★ | ★★★★★ | Kiến trúc mới nhất |

## Cấu trúc thư mục sau training

```
runs/
└── detect/
    └── cad_yolo/
        ├── weights/
        │   ├── best.pt      ← model tốt nhất (dùng cho deploy)
        │   └── last.pt      ← checkpoint cuối (dùng để resume)
        ├── results.png      ← biểu đồ loss/mAP
        ├── confusion_matrix.png
        ├── PR_curve.png
        ├── F1_curve.png
        └── val_batch*.jpg   ← ví dụ predict trên val set
```

## Tips

- **Dataset nhỏ (813 ảnh)**: dùng `--augment` để tăng hiệu quả regularization
- **Ảnh bản vẽ CAD thường rất to**: `--imgsz 1280` hoặc `1920` phù hợp hơn 640
- **Mất điện / gián đoạn**: resume bằng `--resume runs/.../weights/last.pt`
- **Overfitting**: giảm `--epochs`, tăng `--patience`, thêm `--augment`
- **GPU VRAM thấp**: giảm `--batch` (e.g. `--batch 4`) hoặc `--imgsz 640`
