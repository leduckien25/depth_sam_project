# DepthSAM (bản tự cài đặt) – Phát hiện vật ngụy trang bằng độ sâu

Bài tập lớn môn Xử lý ảnh. Nhóm tự viết lại (bản đơn giản hoá) mô hình trong bài báo:

> *Beyond Appearance: Camouflaged Object Detection via Geometric Structure* (DepthSAM, CVPR 2026).
> Code gốc của tác giả: <https://github.com/hjy0518/DepthSAM>

Đây **không phải** code chính thức. Code được viết tối giản để dễ hiểu và dễ trình bày.

---

## 1. Ý tưởng

Vật ngụy trang có màu và vân giống nền, nhưng **hình khối 3D (độ sâu)** của vật vẫn khác nền.
Vì vậy mô hình dùng **Depth Anything V2 (DAv2)**, một mô hình ước lượng độ sâu đã train sẵn, làm xương sống. Sau đó gắn thêm 2 module học được:

| Module | Tên đầy đủ | File | Vai trò |
|---|---|---|---|
| **SMEA** | Sparse Mixture-of-Experts Adapter | `src/smea.py` | Đặt **trước mỗi block** (24 block) của encoder DAv2 (đã đóng băng). Có 8 chuyên gia, mỗi lần chọn top-2. `x_m = x + Σ gᵢ·Eᵢ(x)` (công thức 1) |
| **GSFM** | Geometric–Semantic Fusion Module | `src/gsfm.py` | Trộn **nối tầng** 4 tầng đặc trưng của decoder DPT: `Fo = SFRM(SFRM(SFRM(F4, F3), F2), F1)` (công thức 2) |
| **SFRM** | Spatial-Frequency Refinement Module | `src/sfrm.py` | Nhận 2 đầu vào (tầng sâu và tầng nông). Có 2 luồng: **không gian** (conv 3×3) và **tần số** (FFT → conv → IFFT) |

Luồng dữ liệu (ảnh 378×378):
```
Ảnh → [DAv2 encoder ViT-L (đóng băng) + SMEA (học)]
    → [DAv2 decoder DPT (đóng băng)] → F4 27×27, F3 54×54, F2 108×108, F1 216×216 (256 kênh)
    → [GSFM = 3 khối SFRM (học)] → [pred_head (học)] → sigmoid → mask 0..1
```
Tham số được học: **16.088.513** (chỉ SMEA, GSFM và pred_head).

**Chỗ đơn giản hơn bài báo** (cố ý, để dễ hiểu):
- SFRM chỉ dùng conv, không có 4 nhánh attention.
- Không có aux loss cho MoE, không có SAM mask decoder.
- Ảnh 378 px thay vì 384 px, vì DAv2 bắt buộc kích thước chia hết cho 14 (378 = 27×14).

---

## 2. Kết quả

Train trên CAMO + COD10K (4040 ảnh), 20 epoch, GPU Kaggle T4. Kết quả đo trên tập test COD10K (2026 ảnh):

| | Sα ↑ | MAE ↓ |
|---|---|---|
| Lần 1: BCE + IoU | 0.8963 | 0.0180 |
| **Lần 2: structure loss** (file mô hình đang dùng) | **0.9046** | **0.0179** |
| Bài báo DepthSAM-L, 384 px | 0.905 | 0.017 |

Lưu ý: file `best` được chọn theo Sα trên 500 ảnh đầu của tập test COD10K, nên số liệu có thể hơi lạc quan.

---

## 3. Cài đặt

```bash
pip install torch torchvision opencv-python pillow numpy pysodmetrics
```

### Lấy file mô hình đã train (không cần train lại)
1. Tải `depthsam_best_structure.pth` và `depth_anything_v2_vitl.pth`
2. Đặt vào thư mục `weights/`.

Chạy demo và đo điểm **không cần** file `depth_anything_v2_vitl.pth`. File đó chỉ cần khi train lại.

---

## 4. Cách chạy

### Demo 1 ảnh (chạy được trên CPU)
Sửa đường dẫn ảnh ở **dòng 19** của `inference.py` (mặc định là `test/gecko.png`), rồi chạy:
```bash
python inference.py
```
Đầu ra:
- `pred_raw_map.png`: bản đồ xác suất (dùng để chấm điểm)
- `pred_binary_raw.png`: mask nhị phân, ngưỡng 0.5
- `comparison_raw.png`: ghép ảnh gốc | bản đồ xác suất | mask

### Đo 4 chỉ số (Sα, Fωβ, MAE, mean Eξ) trên CAMO, COD10K, NC4K
```bash
python evaluate_all.py              # cả 3 tập
python evaluate_all.py CAMO         # 1 tập
python evaluate_all.py CAMO 20      # 20 ảnh đầu, để thử nhanh
```
Trên CPU mất khoảng 14 giây/ảnh (cả 3 tập khoảng 24 giờ), nên dùng GPU.
Có thể đổi file mô hình và thư mục dữ liệu bằng biến môi trường `CHECKPOINT` và `DATA_ROOT`.

### Train lại (cần GPU khoảng 10 GB, ví dụ Kaggle T4)
Chỉ cần khi muốn train một mô hình mới (ví dụ đổi loss hoặc đổi kích thước ảnh). Chạy demo hay đo điểm thì **không cần** bước này.

1. Tải trọng số DAv2 đã train sẵn vào `weights/`. Phần DAv2 bị đóng băng (không học), nên phải nạp sẵn trọng số này; nếu không, nó chỉ là số ngẫu nhiên:
   <https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth>
2. Chạy:
```bash
# Linux / Kaggle
EPOCHS=20 DATA_ROOT=dataset python train.py
```
```powershell
# Windows PowerShell
$env:EPOCHS = "20"; $env:DATA_ROOT = "dataset"; python train.py
```
Mỗi epoch mất khoảng 26 phút trên T4 (gồm cả bước chấm 500 ảnh test).
Script lưu `weights/depthsam_last.pth` (sau mỗi epoch) và `weights/depthsam_best.pth` (theo Sα cao nhất).

---

## 5. Cấu trúc thư mục

```text
depth_sam_project/
├── dataset/                        # không đưa lên git
│   ├── TrainDataset/{Imgs,GT}/     # CAMO + COD10K train: 4040 ảnh
│   └── TestDataset/
│       ├── CAMO/{Imgs,GT}/         # 250 ảnh
│       ├── COD10K/{Imgs,GT}/       # 2026 ảnh
│       └── NC4K/{Imgs,GT}/         # 4121 ảnh
├── depth_anything_v2/              # code DAv2 chính thức, không sửa
├── src/
│   ├── dataset.py                  # đọc ảnh + mask, resize 378, augment (lật ngang, crop)
│   ├── smea.py                     # SMEA + SMEABlock
│   ├── sfrm.py                     # SFRM: luồng không gian + luồng tần số
│   ├── gsfm.py                     # GSFM: 3 SFRM nối tầng
│   └── model.py                    # ghép DAv2 + SMEA + GSFM + pred_head
├── test/                           # ảnh mẫu để demo
├── weights/                        # file .pth, không đưa lên git
├── train.py                        # train (structure loss, AMP fp16, AdamW lr 1e-4, batch 2)
├── inference.py                    # demo 1 ảnh
└── evaluate_all.py                 # đo 4 chỉ số trên 3 tập test
```
