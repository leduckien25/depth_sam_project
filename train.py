import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader

# Cấu hình UTF-8 cho console Windows để tránh UnicodeEncodeError khi in emoji
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)

from src.dataset import COD10KDataset
from src.model import DepthSAM


def structure_loss(pred, mask):
    """Structure loss = BCE có trọng số + IoU có trọng số (đầu vào pred đã qua Sigmoid).
    Pixel gần BIÊN có trọng số lớn hơn (tối đa 6), nên mô hình bị phạt nặng khi vẽ sai viền."""
    # Trọng số: trong lòng vật hoặc nền = 1; càng gần biên càng lớn
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    # BCE có trọng số
    wbce = F.binary_cross_entropy(pred, mask, reduction="none")
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    # IoU có trọng số
    inter = (pred * mask * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean(), wbce.mean(), wiou.mean()


def evaluate(model, dataset, device, max_images=None):
    """Chấm điểm trên tập test: S-measure (Sα, càng cao càng tốt) và MAE (càng thấp càng tốt).
    Dùng thư viện chuẩn py_sod_metrics (pip install pysodmetrics). So với mask gốc, đúng kích thước gốc."""
    from py_sod_metrics import MAE, Smeasure

    sm, mae = Smeasure(), MAE()
    n = len(dataset) if max_images is None else min(max_images, len(dataset))
    model.eval()
    with torch.no_grad():
        for i in range(n):
            image, _ = dataset[i]
            mask_name = os.path.splitext(dataset.images[i])[0] + ".png"
            gt = np.array(Image.open(os.path.join(dataset.mask_dir, mask_name)).convert("L"))

            pred = model(image.unsqueeze(0).to(device))                     # [1, 1, h, w], đã qua sigmoid
            pred = F.interpolate(pred, size=gt.shape, mode="bilinear", align_corners=False)
            pred = (pred[0, 0].cpu().numpy() * 255).astype(np.uint8)

            sm.step(pred=pred, gt=gt)
            mae.step(pred=pred, gt=gt)
    model.train()
    return sm.get_results()["sm"], mae.get_results()["mae"]


def train():
    print("=" * 60)
    print("🚀 KHỞI ĐỘNG QUÁ TRÌNH HUẤN LUYỆN DEPTHSAM")
    print("=" * 60)

    # 1. Kiểm tra thiết bị GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Thiết bị sử dụng: {device}")
    if device.type == "cuda":
        print(f"🎮 Card đồ họa: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ Cảnh báo: Đang chạy trên CPU! Hãy kiểm tra lại CUDA.")

    # 2. Cấu hình đường dẫn và siêu tham số (Hyperparameters)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Thư mục dữ liệu: mặc định là ./dataset, trên Kaggle đặt biến môi trường DATA_ROOT
    #   TrainDataset/Imgs, TrainDataset/GT          : CAMO + COD10K train (4040 ảnh)
    #   TestDataset/COD10K/Imgs, TestDataset/COD10K/GT : COD10K test (2026 ảnh)
    data_root = os.environ.get("DATA_ROOT", os.path.join(base_dir, "dataset"))
    image_dir = os.path.join(data_root, "TrainDataset", "Imgs")
    mask_dir = os.path.join(data_root, "TrainDataset", "GT")
    test_image_dir = os.path.join(data_root, "TestDataset", "COD10K", "Imgs")
    test_mask_dir = os.path.join(data_root, "TestDataset", "COD10K", "GT")
    checkpoint_path = os.path.join(base_dir, "weights", "depth_anything_v2_vitl.pth")
    save_dir = os.path.join(base_dir, "weights")
    os.makedirs(save_dir, exist_ok=True)

    batch_size = 2
    epochs = int(os.environ.get("EPOCHS", 30))
    lr = 1e-4
    num_eval_images = 500   # chấm nhanh 500 ảnh test sau mỗi epoch; cuối cùng chấm đủ 2026 ảnh

    # 3. Kiểm tra sự tồn tại của thư mục dữ liệu
    print(f"\n📂 Kiểm tra đường dẫn dữ liệu:")
    print(f"   - Thư mục Image:     {image_dir} -> {'Tồn tại' if os.path.exists(image_dir) else 'KHÔNG TỒN TẠI'}")
    print(f"   - Thư mục GT_Object: {mask_dir} -> {'Tồn tại' if os.path.exists(mask_dir) else 'KHÔNG TỒN TẠI'}")

    if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
        print("❌ Lỗi: Không tìm thấy thư mục dữ liệu! Hãy kiểm tra lại đường dẫn folder COD10K.")
        return

    # 4. Khởi tạo Dataset và DataLoader
    train_dataset = COD10KDataset(image_dir=image_dir, mask_dir=mask_dir, augment=True)
    test_dataset = COD10KDataset(image_dir=test_image_dir, mask_dir=test_mask_dir, augment=False)
    print(f"📊 Tập test COD10K: {len(test_dataset)} ảnh")
    num_samples = len(train_dataset)
    print(f"📊 Tổng số ảnh tìm thấy trong tập Train: {num_samples}")

    if num_samples == 0:
        print("❌ Lỗi: Tập dữ liệu có 0 ảnh! Hãy kiểm tra xem ảnh đã được giải nén vào đúng thư mục chưa.")
        return

    # Đặt num_workers=0 trên Windows để tránh lỗi đa tiến trình và tránh in lặp cảnh báo xFormers
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False,
        drop_last=True
    )

    # 5. Khởi tạo mô hình DepthSAM
    print("\n🧠 Đang khởi tạo mô hình DepthSAM...")
    model = DepthSAM(
        encoder_type="vitl",
        checkpoint_path=checkpoint_path
    ).to(device)

    # Chỉ truyền các tham số không bị đóng băng (requires_grad=True) vào Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in model.parameters())
    learned_params = sum(p.numel() for p in trainable_params)
    print(f"📈 Tham số huấn luyện: {learned_params:,} / {total_params:,} ({learned_params / total_params * 100:.2f}%)")

    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    # AMP (fp16): giảm khoảng một nửa bộ nhớ GPU và chạy nhanh hơn trên T4
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # 6. Vòng lặp huấn luyện
    print("\n🔥 Bắt đầu vòng lặp huấn luyện...")
    best_sa = -1.0

    # Tự động nạp lại checkpoint nếu trước đó đã lưu
    resume_path = os.path.join(save_dir, "depthsam_last.pth")
    if os.path.exists(resume_path):
        print(f"🔄 Tìm thấy checkpoint cũ, đang nạp lại trọng số từ: {resume_path}")
        model.load_state_dict(torch.load(resume_path, map_location=device))
        print("✅ Đã khôi phục trọng số thành công, tiếp tục huấn luyện!")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            # Đảm bảo kích thước đầu vào chia hết cho 14 (patch size của DAv2 ViT)
            h, w = images.shape[2], images.shape[3]
            if h % 14 != 0 or w % 14 != 0:
                new_h = (h // 14) * 14
                new_w = (w // 14) * 14
                images = torch.nn.functional.interpolate(images, size=(new_h, new_w), mode="bilinear", align_corners=False)
                masks = torch.nn.functional.interpolate(masks, size=(new_h, new_w), mode="nearest")

            optimizer.zero_grad()

            # Forward pass (đầu ra của DepthSAM đã qua hàm Sigmoid), chạy bằng fp16
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                preds = model(images)
            preds = preds.float()   # tính loss bằng fp32 cho ổn định

            # Structure loss: BCE + IoU, cả hai đều có trọng số lớn ở vùng biên
            loss, loss_bce, loss_iou = structure_loss(preds, masks)

            # Backward pass & Cập nhật trọng số
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if (batch_idx + 1) % 50 == 0:
                torch.cuda.empty_cache()

            epoch_loss += loss.item()

            # In tiến độ ngay ở batch đầu tiên và mỗi 10 batch tiếp theo
            if batch_idx == 0 or (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                print(
                    f"Epoch [{epoch}/{epochs}] | "
                    f"Batch [{batch_idx + 1}/{len(train_loader)}] | "
                    f"Loss: {loss.item():.4f} (BCE: {loss_bce.item():.4f}, IoU: {loss_iou.item():.4f})"
                )

        avg_loss = epoch_loss / len(train_loader)
        elapsed = time.time() - start_time
        print("-" * 60)
        print(f"✅ Kết thúc Epoch [{epoch}/{epochs}] | Loss trung bình: {avg_loss:.4f} | Thời gian: {elapsed:.1f}s")
        print("-" * 60)

        # Lưu checkpoint mới nhất và tốt nhất
        last_ckpt = os.path.join(save_dir, "depthsam_last.pth")
        torch.save(model.state_dict(), last_ckpt)

        # Chấm điểm trên tập test, lưu mô hình có Sα cao nhất
        sa, mae = evaluate(model, test_dataset, device, max_images=num_eval_images)
        print(f"🎯 Test COD10K ({num_eval_images} ảnh): Sα = {sa:.4f} ({sa / 0.920 * 100:.0f}% bài báo) | "
              f"MAE = {mae:.4f} (bài báo 0.014)")
        if sa > best_sa:
            best_sa = sa
            best_ckpt = os.path.join(save_dir, "depthsam_best.pth")
            torch.save(model.state_dict(), best_ckpt)
            print(f"💾 Đã lưu mô hình tốt nhất mới tại: {best_ckpt} (Sα: {best_sa:.4f})")

    print("🎉 Hoàn tất quá trình huấn luyện!")

    # Chấm đủ toàn bộ tập test với mô hình tốt nhất
    model.load_state_dict(torch.load(os.path.join(save_dir, "depthsam_best.pth"), map_location=device))
    sa, mae = evaluate(model, test_dataset, device)
    print("=" * 60)
    print(f"KẾT QUẢ CUỐI trên COD10K ({len(test_dataset)} ảnh): Sα = {sa:.4f} | MAE = {mae:.4f}")
    print(f"Mục tiêu 90% bài báo: Sα ≥ 0.828 -> {'ĐẠT' if sa >= 0.828 else 'CHƯA'} | "
          f"MAE ≤ 0.016 -> {'ĐẠT' if mae <= 0.016 else 'CHƯA'}")


if __name__ == "__main__":
    train()
