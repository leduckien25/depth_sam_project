import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Cấu hình UTF-8 cho console Windows để tránh UnicodeEncodeError khi in emoji
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)

from src.dataset import COD10KDataset
from src.model import DepthSAM


def iou_loss(preds, targets, smooth=1e-6):
    """Hàm tính IoU Loss cho bài toán phân đoạn vật thể ngụy trang (đầu vào đã qua Sigmoid)."""
    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    total = (preds + targets).sum()
    union = total - intersection
    iou = (intersection + smooth) / (union + smooth)
    return 1.0 - iou


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
    image_dir = os.path.join(base_dir, "dataset", "COD10K-v3", "Image")
    mask_dir = os.path.join(base_dir, "dataset", "COD10K-v3", "GT_Object")
    checkpoint_path = os.path.join(base_dir, "weights", "depth_anything_v2_vitl.pth")
    save_dir = os.path.join(base_dir, "weights")
    os.makedirs(save_dir, exist_ok=True)

    batch_size = 2
    epochs = 20
    lr = 1e-4

    # 3. Kiểm tra sự tồn tại của thư mục dữ liệu
    print(f"\n📂 Kiểm tra đường dẫn dữ liệu:")
    print(f"   - Thư mục Image:     {image_dir} -> {'Tồn tại' if os.path.exists(image_dir) else 'KHÔNG TỒN TẠI'}")
    print(f"   - Thư mục GT_Object: {mask_dir} -> {'Tồn tại' if os.path.exists(mask_dir) else 'KHÔNG TỒN TẠI'}")

    if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
        print("❌ Lỗi: Không tìm thấy thư mục dữ liệu! Hãy kiểm tra lại đường dẫn folder COD10K.")
        return

    # 4. Khởi tạo Dataset và DataLoader
    train_dataset = COD10KDataset(image_dir=image_dir, mask_dir=mask_dir)
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
    bce_criterion = nn.BCELoss()

    # 6. Vòng lặp huấn luyện
    print("\n🔥 Bắt đầu vòng lặp huấn luyện...")
    best_loss = float("inf")

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

            # Forward pass (đầu ra của DepthSAM đã qua hàm Sigmoid)
            preds = model(images)

            # Tính tổng hợp BCE Loss + IoU Loss
            loss_bce = bce_criterion(preds, masks)
            loss_iou = iou_loss(preds, masks)
            loss = loss_bce + loss_iou

            # Backward pass & Cập nhật trọng số
            loss.backward()
            optimizer.step()

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

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_ckpt = os.path.join(save_dir, "depthsam_best.pth")
            torch.save(model.state_dict(), best_ckpt)
            print(f"💾 Đã lưu mô hình tốt nhất mới tại: {best_ckpt} (Loss: {best_loss:.4f})")

    print("🎉 Hoàn tất quá trình huấn luyện!")


if __name__ == "__main__":
    train()
