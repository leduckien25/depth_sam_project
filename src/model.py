import torch
import torch.nn as nn
import torch.nn.functional as F

from src.smea import SparseMoEAdapter
from src.gsfm import GSFM

class DepthSAM(nn.Module):
    def __init__(self, encoder_channels=[48, 96, 192, 384]):
        super().__init__()
        
        # 1. Mạng nền Encoder (ví dụ DAv2-Small với các mức channel tương ứng)
        # Trong thực tế, bạn sẽ khởi tạo backbone DAv2 ở đây và đóng băng (freeze) nó lại.
        # Tạm thời ta dùng cấu trúc chuẩn channel của DAv2-Small: [48, 96, 192, 384]
        self.channel_list = encoder_channels
        
        # 2. Nhúng các cục SMEA Adapter vào từng tầng của Encoder
        self.smea_adapters = nn.ModuleList([
            SparseMoEAdapter(dim=c, num_experts=4, top_k=2) for c in self.channel_list
        ])
        
        # 3. Phần Decoder sử dụng GSFM để dung hợp không gian - tần số
        self.decoder_gsfm = GSFM(channel_list=self.channel_list)
        
        # 4. Standard Prediction Head (Lớp tích chập tiêu chuẩn đúng chuẩn bài báo)
        # Làm nhiệm vụ nén từ số kênh của GSFM về 1 kênh duy nhất để tạo mặt nạ (Mask)
        self.pred_head = nn.Sequential(
            nn.Conv2d(self.channel_list[-1], 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1)
        )

    def forward(self, x, backbone_features=None):
        """
        Args:
            x: Ảnh đầu vào [Batch, 3, H, W]
            backbone_features: Danh sách 4 tầng đặc trưng lấy trực tiếp từ DAv2 Encoder (nếu có sẵn)
        """
        original_size = x.shape[2:]
        
        # Nếu chưa truyền features từ ngoài vào, ta giả lập luồng qua encoder 
        # (Sau này khi cắm DAv2 thật vào, bạn chỉ cần gọi self.backbone(x) để lấy danh sách 4 tầng features)
        features = backbone_features if backbone_features is not None else [
            torch.randn(x.size(0), self.channel_list[0], x.size(2)//2, x.size(3)//2, device=x.device),
            torch.randn(x.size(0), self.channel_list[1], x.size(2)//4, x.size(3)//4, device=x.device),
            torch.randn(x.size(0), self.channel_list[2], x.size(2)//8, x.size(3)//8, device=x.device),
            torch.randn(x.size(0), self.channel_list[3], x.size(2)//16, x.size(3)//16, device=x.device),
        ]
        
        # --- BƯỚC 1: ĐI QUA SMEA ADAPTERS TẠI ENCODER ---
        refined_features = []
        for feat, smea in zip(features, self.smea_adapters):
            b, c, h, w = feat.shape
            # Chuyển [B, C, H, W] thành dạng token [B, H*W, C] để đưa vào cơ chế MoE của SMEA
            flat_feat = feat.flatten(2).transpose(1, 2)
            refined_flat = smea(flat_feat)
            # Đổi ngược lại hình học [B, C, H, W]
            refined_feat = refined_flat.transpose(1, 2).reshape(b, c, h, w)
            refined_features.append(refined_feat)
            
        # --- BƯỚC 2: ĐI QUA GSFM TẠI DECODER ---
        decoded_feat = self.decoder_gsfm(refined_features)
        
        # --- BƯỚC 3: QUA HEAD ĐỂ RA KẾT QUẢ CUỐI CÙNG ---
        out = self.pred_head(decoded_feat)
        
        # Phóng to mặt nạ về đúng kích thước ảnh gốc ban đầu
        out = F.interpolate(out, size=original_size, mode='bilinear', align_corners=False)
        
        # Dùng sigmoid để ép xác suất pixel thuộc về vật thể ngụy trang về khoảng [0, 1]
        return torch.sigmoid(out)