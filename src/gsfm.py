import torch
import torch.nn as nn
import torch.nn.functional as F

from src.sfrm import SpatialFrequencyModule

class GSFM(nn.Module):
    def __init__(self, channel_list=[256, 256, 256, 256]):
        super().__init__()
        # Tạo danh sách các khối SFRM cho từng tầng đặc trưng của Decoder
        self.sfrm_blocks = nn.ModuleList([
            SpatialFrequencyModule(c) for c in channel_list
        ])
        # Lớp gộp cuối cùng (Fusion Conv)
        self.fusion_conv = nn.Conv2d(sum(channel_list), channel_list[-1], kernel_size=1)

    def forward(self, features):
        # features là danh sách các tầng đặc trưng từ Decoder [F1, F2, F3, F4]
        refined_features = []
        for feat, block in zip(features, self.sfrm_blocks):
            refined_features.append(block(feat))
            
        # Đồng bộ kích thước không gian của các tầng về cùng một cỡ để gộp lại
        target_size = refined_features[0].shape[2:]
        resized_feats = [
            F.interpolate(f, size=target_size, mode='bilinear', align_corners=False) 
            for f in refined_features
        ]

        # Gộp các đặc trưng đã được đồng bộ kích thước lại với nhau
        concat_feat = torch.cat(resized_feats, dim=1)

        return self.fusion_conv(concat_feat)