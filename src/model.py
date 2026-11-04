import torch
import torch.nn as nn
import torch.nn.functional as F

from depth_anything_v2.dpt import DepthAnythingV2
from src.smea import SparseMoEAdapter
from src.gsfm import GSFM


class DepthSAM(nn.Module):
    def __init__(self, encoder_type="vitl", checkpoint_path="weights/depth_anything_v2_vitl.pth"):
        super().__init__()

        # Cấu hình chuẩn tương ứng cho từng encoder của DAv2
        model_configs = {
            "vits": {"features": 64,  "out_channels": [48, 96, 192, 384]},
            "vitb": {"features": 128, "out_channels": [96, 192, 384, 768]},
            "vitl": {"features": 256, "out_channels": [256, 512, 1024, 1024]},
        }

        if encoder_type not in model_configs:
            raise ValueError(f"Không hỗ trợ encoder: {encoder_type}")

        cfg = model_configs[encoder_type]
        self.channel_list = cfg["out_channels"]

        # 1. Khởi tạo backbone DAv2
        print(f"🔄 Đang tải backbone DAv2 ({encoder_type})...")
        self.backbone = DepthAnythingV2(
            encoder=encoder_type,
            features=cfg["features"],
            out_channels=self.channel_list,
        )

        if checkpoint_path:
            self.backbone.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
            print(f"✅ Đã nạp weights DAv2 thành công từ: {checkpoint_path}")

        # Đóng băng toàn bộ tham số của DAv2
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 2. SMEA Adapters
        self.smea_adapters = nn.ModuleList([
            SparseMoEAdapter(dim=c, num_experts=4, top_k=2) for c in self.channel_list
        ])

        # 3. Decoder GSFM
        self.decoder_gsfm = GSFM(channel_list=self.channel_list)

        # 4. Prediction Head
        self.pred_head = nn.Sequential(
            nn.Conv2d(self.channel_list[-1], 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
        )

    def forward(self, x):
        original_size = x.shape[2:]
        b, _, h, w = x.shape
        patch_h, patch_w = h // 14, w // 14

        # 1. Trích xuất đặc trưng từ ViT encoder
        layer_indices = self.backbone.intermediate_layer_idx[self.backbone.encoder]
        raw_features = self.backbone.pretrained.get_intermediate_layers(
            x,
            layer_indices,
            return_class_token=True,
        )

        # 2. Chiếu số kênh qua depth_head.projects của DAv2
        features = []
        for i, feat_tuple in enumerate(raw_features):
            patch_tokens = feat_tuple[0]  # [B, N, 1024]
            feat_map = patch_tokens.permute(0, 2, 1).reshape(b, -1, patch_h, patch_w)
            
            # Gọi qua depth_head.projects để hạ từ 1024 về [256, 512, 1024, 1024]
            feat_map = self.backbone.depth_head.projects[i](feat_map)
            features.append(feat_map)

        # 3. Đi qua SMEA Adapters
        refined_features = []
        for feat, smea in zip(features, self.smea_adapters):
            bc, c_dim, ch, cw = feat.shape
            flat_feat = feat.flatten(2).transpose(1, 2)
            refined_flat = smea(flat_feat)
            refined_feat = refined_flat.transpose(1, 2).reshape(bc, c_dim, ch, cw)
            refined_features.append(refined_feat)

        # 4. Dung hợp qua Decoder GSFM & Head
        decoded_feat = self.decoder_gsfm(refined_features)
        out = self.pred_head(decoded_feat)
        out = F.interpolate(out, size=original_size, mode="bilinear", align_corners=False)

        return torch.sigmoid(out)
