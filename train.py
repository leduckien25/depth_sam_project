import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.dataset import COD10KDataset
from src.model import DepthSAM

def iou_loss(pred, target):
    """Tính toán IoU Loss để ép mô hình bắt trọn vẹn hình thù khối của vật thể ngụy trang"""
    inter = (pred * target).sum(dim=(2, 3))
    union = (pred + target).sum(dim=(2, 3)) - inter
    iou = 1.0 - (inter + 1e-5) / (union + 1e-5)
    return iou.mean()

def structure_loss(pred, mask):
    """
    Hybrid Loss chuẩn cho COD: 
    Kết hợp Binary Cross Entropy (chuẩn hóa pixel) + IoU Loss (chuẩn hóa hình khối)
    """
    bce = F.binary_cross_entropy(pred, mask, reduction='mean')
    iou = iou_loss(pred, mask)
    return bce + iou

def train():
    # 1. Cấu hình thiết bị phần cứng (GPU nếu có, không thì dùng CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Đang sử dụng thiết bị: {device}")
    
    # Hyperparameters (Cấu hình huấn luyện)
    batch_size = 4        # Tùy chỉnh theo VRAM của card RTX 4050 trên laptop của bạn
    epochs = 20           # Số vòng lặp học
    learning_rate = 1e-4  # Tốc độ học khởi điểm chuẩn cho fine-tune adapter
    
    # Đường dẫn tới thư mục dữ liệu COD10K (Bạn điều chỉnh lại nếu cấu trúc khác nhé)
    image_dir = "./data/COD10K/Train/Image"
    mask_dir = "./data/COD10K/Train/Mask"
    
    # 2. Khởi tạo Dataset và DataLoader
    print("📂 Đang tải tập dữ liệu COD10K...")
    train_dataset = COD10KDataset(image_dir=image_dir, mask_dir=mask_dir)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2, 
        drop_last=True
    )
    
    # 3. Khởi tạo mô hình DepthSAM hoàn chỉnh
    # (Đảm bảo bạn đã đặt file trọng số depth_anything_v2_vits.pth vào thư mục weights/)
    print("🧠 Đang khởi tạo mô hình DepthSAM...")
    model = DepthSAM(encoder_type='vits', checkpoint_path='weights/depth_anything_v2_vits.pth').to(device)
    
    # 4. Cấu hình Optimizer
    # ⚠️ QUAN TRỌNG: Chỉ truyền vào optimizer những tham số có yêu cầu đạo hàm (requires_grad = True),
    # điều này giúp bảo vệ tuyệt đối phần backbone DAv2 đã đóng băng.
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=1e-4)
    
    # Tạo thư mục chứa weights nếu chưa có
    os.makedirs("./weights", exist_ok=True)
    
    # 5. Vòng lặp Train chính (Training Loop)
    print("🔥 Bắt đầu vòng lặp huấn luyện DepthSAM...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)
            
            # Bước a: Forward pass (Mô hình dự đoán mặt nạ)
            preds = model(images)
            
            # Bước b: Tính sai số bằng Hybrid Loss (BCE + IoU)
            loss = structure_loss(preds, masks)
            
            # Bước c: Backpropagation và cập nhật trọng số cho SMEA, GSFM, Pred Head
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # In thông tin tiến trình từng batch
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Batch [{batch_idx}/{len(train_loader)}] | Loss: {loss.item():.4f}")
                
        # Tổng kết sau mỗi epoch
        avg_loss = epoch_loss / len(train_loader)
        print(f"🎉 KẾT THÚC EPOCH {epoch+1} | Loss trung bình: {avg_loss:.4f}")
        
        # Lưu file checkpoint weights sau mỗi epoch
        checkpoint_path = f"./weights/depthsam_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"💾 Đã lưu checkpoint tại: {checkpoint_path}\n")

if __name__ == "__main__":
    train()