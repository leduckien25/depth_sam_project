import torch
import torch.nn as nn
import torch.nn.functional as F

from depth_anything_v2.dpt import DepthAnythingV2
from src.smea import SparseMoEAdapter
from src.gsfm import GSFM

class DepthSAM(nn.Module):
    def __init__(self, encoder_type='vits', checkpoint_path='weights/depth_anything_v2_vitl.pth'):
        super().__init__()
        
        # 1. Khai báo số kênh tương ứng cho từng loại encoder của DAv2
        if encoder_type == 'vits':
            self.channel_list = [48, 96, 192, 384]
        elif encoder_type == 'vitb':
            self.channel_list = [96, 192, 384, 768]
        elif encoder_type == 'vitl':
            self.channel_list = [256, 512, 1024, 1024]
        else:
            raise ValueError(f"Không hỗ trợ encoder: {encoder_type}")
            
        # ==========================================
        # 2. KHỞI TẠO MẠNG NỀN DAV2 VÀ ĐÓNG BĂNG (FREEZE)
        # ==========================================
        print(f"🔄 Đang tải backbone DAv2 ({encoder_type})...")
        self.backbone = DepthAnythingV2(
            encoder=encoder_type, 
            features=64, 
            out_channels=self.channel_list
        )
        
        if checkpoint_path:
            self.backbone.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
            print(f"✅ Đã nạp weights DAv2 thành công từ: {checkpoint_path}")
            
        # Đóng băng toàn bộ encoder để giữ nguyên tri thức nền của DAv2
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # 3. Các cục SMEA Adapter gắn vào các tầng của Encoder
        self.smea_adapters = nn.ModuleList([
            SparseMoEAdapter(dim=c, num_experts=4, top_k=2) for c in self.channel_list
        ])
        
        # 4. Decoder GSFM (chứa các khối SFRM)
        self.decoder_gsfm = GSFM(channel_list=self.channel_list)
        
        # 5. Prediction Head chuẩn (nén kênh về 1 để tạo Mask ngụy trang)
        self.pred_head = nn.Sequential(
            nn.Conv2d(self.channel_list[-1], 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1)
        )

    def forward(self, x):
        original_size = x.shape[2:]
        
        # =========================================================================
        # BƯỚC 1: LẤY CÁC TẦNG ĐẶC TRƯNG (MULTI-LEVEL FEATURES) TỪ DAv2 THẬT
        # =========================================================================
        # Hàm forward_features hoặc get_intermediate_layers của DAv2 sẽ trả về list 4 tầng đặc trưng
        features = self.backbone.get_intermediate_layers(x)
        
        # =========================================================================
        # BƯỚC 2: ĐI QUA SMEA ADAPTERS TẠI ENCODER
        # =========================================================================
        refined_features = []
        for feat, smea in zip(features, self.smea_adapters):
            b, c, h, w = feat.shape
            # Chuyển đổi tensor [B, C, H, W] sang dạng token [B, H*W, C] cho cơ chế MoE
            flat_feat = feat.flatten(2).transpose(1, 2)
            refined_flat = smea(flat_feat)
            # Khôi phục lại hình học [B, C, H, W]
            refined_feat = refined_flat.transpose(1, 2).reshape(b, c, h, w)
            refined_features.append(refined_feat)
            
        # =========================================================================
        # BƯỚC 3: DUNG HỢP QUA GSFM TẠI DECODER
        # =========================================================================
        decoded_feat = self.decoder_gsfm(refined_features)
        
        # =========================================================================
        # BƯỚC 4: XUẤT MẶT NẠ NGỤY TRANG QUA HEAD
        # =========================================================================
        out = self.pred_head(decoded_feat)
        out = F.interpolate(out, size=original_size, mode='bilinear', align_corners=False)
        
        return torch.sigmoid(out)