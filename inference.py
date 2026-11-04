import os
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from src.model import DepthSAM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Nạp checkpoint DepthSAM tốt nhất
checkpoint_path = r"weights/depthsam_best.pth"
model = DepthSAM(encoder_type="vitl", checkpoint_path=None).to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# 2. Xử lý ảnh đầu vào
input_image_path = r"test/gecko.png"

raw_img = Image.open(input_image_path).convert("RGB")
orig_w, orig_h = raw_img.size

# Bo tròn kích thước về bội số gần nhất của 14 (chuẩn patch size ViT)
target_h = min(518, (orig_h // 14) * 14)
target_w = min(518, (orig_w // 14) * 14)

transform = transforms.Compose([
    transforms.Resize((target_h, target_w)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

input_tensor = transform(raw_img).unsqueeze(0).to(device)

# 3. Dự đoán trực tiếp từ mạng (End-to-End, thuần PyTorch)
with torch.no_grad():
    pred = model(input_tensor)
    pred = F.interpolate(pred, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
    prob_map = pred.squeeze().cpu().numpy()

# 4. Xuất kết quả nguyên bản (Raw outputs)

# 4.1. Bản đồ xác suất liên tục (Continuous Probability Map / Saliency Map)
# Đây là định dạng chuẩn để ném vào các bộ benchmark tính MAE, S-measure, E-measure
raw_prob_vis = (prob_map * 255).astype(np.uint8)
cv2.imwrite("pred_raw_map.png", raw_prob_vis)

# 4.2. Mặt nạ nhị phân tiêu chuẩn (Cắt ngưỡng 0.5 mặc định, không lọc nhiễu)
binary_mask = (prob_map > 0.5).astype(np.uint8) * 255
cv2.imwrite("pred_binary_raw.png", binary_mask)

# 4.3. Ghép dải trực quan cho bài báo: Ảnh gốc | Map xác suất thô | Mask nhị phân chuẩn
orig_cv = cv2.cvtColor(np.array(raw_img), cv2.COLOR_RGB2BGR)
raw_bgr = cv2.cvtColor(raw_prob_vis, cv2.COLOR_GRAY2BGR)
bin_bgr = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)

comparison = np.hstack([orig_cv, raw_bgr, bin_bgr])
cv2.imwrite("comparison_raw.png", comparison)

print("✅ Đã xuất kết quả nguyên bản:")
print(f" - Map xác suất thô (dùng cho Benchmark/Evaluation): pred_raw_map.png")
print(f" - Mask nhị phân chuẩn 0.5 (dùng cho Qualitative Figures): pred_binary_raw.png")
print(f" - Dải ảnh so sánh trực quan: comparison_raw.png")
print(f" - Thống kê độ tự tin: Max = {prob_map.max():.4f} | Min = {prob_map.min():.4f}")