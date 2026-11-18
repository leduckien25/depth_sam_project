import torch
import torch.nn as nn
import torch.nn.functional as F

from depth_anything_v2.dpt import DepthAnythingV2
from src.smea import SparseMoEAdapter, SMEABlock
from src.gsfm import GSFM


class DepthSAM(nn.Module):
    def __init__(self, encoder_type="vitl", checkpoint_path="weights/depth_anything_v2_vitl.pth"):
        super().__init__()

        # Cấu hình chuẩn cho các loại encoder (DINOv2 ViT) của DAv2
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
        # Toàn bộ tham số của DAv2 sẽ không được huấn luyện lại, chỉ SMEA và GSFM mới được huấn luyện
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 2. SMEA Adapters: đặt TRƯỚC mỗi Transformer Block của encoder (bài báo mục 3.2)
        #    (tạo sau bước đóng băng nên tham số SMEA vẫn được train)
        dim = self.backbone.pretrained.embed_dim # Lấy số kênh của embedding từ encoder ViT
        blocks = self.backbone.pretrained.blocks # Lấy danh sách các Transformer Block của encoder ViT
        for i in range(len(blocks)):
            blocks[i] = SMEABlock(SparseMoEAdapter(dim=dim, num_experts=8, top_k=2), blocks[i])

        # 3. Decoder GSFM: F1..F4 của decoder DPT đều có cfg["features"] kênh
        self.decoder_gsfm = GSFM(channels=cfg["features"])

        # 4. Prediction Head
        self.pred_head = nn.Sequential(
            nn.Conv2d(cfg["features"], 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
        )

    # Forward pass: x -> DAv2 -> SMEA -> GSFM -> Head
    # x: [B, 3, H, W] (ảnh RGB đầu vào)
    def forward(self, x):
        original_size = x.shape[2:]
        b, _, h, w = x.shape
        patch_h, patch_w = h // 14, w // 14

        # 1. Trích xuất đặc trưng từ ViT encoder
        # Lấy các layer để trích xuất đặc trưng từ DAv2
        layer_indices = self.backbone.intermediate_layer_idx[self.backbone.encoder]

        # Trích xuất các đặc trưng trung gian từ DAv2 (bao gồm cả token class)
        raw_features = self.backbone.pretrained.get_intermediate_layers(
            x,
            layer_indices,
            return_class_token=True,
        )

        # 2. Đi qua decoder DPT của DAv2 (giống DPTHead.forward trong depth_anything_v2/dpt.py)
        head = self.backbone.depth_head
        layers = []
        for i, feat_tuple in enumerate(raw_features):
            patch_tokens = feat_tuple[0]  # [B, N, 1024]
            feat_map = patch_tokens.permute(0, 2, 1).reshape(b, -1, patch_h, patch_w)
            feat_map = head.projects[i](feat_map)          # hạ số kênh
            feat_map = head.resize_layers[i](feat_map)     # đổi kích thước: x4, x2, x1, x0.5
            layers.append(feat_map)

        layer_1 = head.scratch.layer1_rn(layers[0])
        layer_2 = head.scratch.layer2_rn(layers[1])
        layer_3 = head.scratch.layer3_rn(layers[2])
        layer_4 = head.scratch.layer4_rn(layers[3])

        path_4 = head.scratch.refinenet4(layer_4, size=layer_3.shape[2:])
        path_3 = head.scratch.refinenet3(path_4, layer_3, size=layer_2.shape[2:])
        path_2 = head.scratch.refinenet2(path_3, layer_2, size=layer_1.shape[2:])
        path_1 = head.scratch.refinenet1(path_2, layer_1)

        # 3. F1..F4 = path_1..path_4: 4 kích thước khác nhau (F1 lớn nhất, F4 nhỏ nhất)
        features = [path_1, path_2, path_3, path_4]

        # 4. Dung hợp qua Decoder GSFM & Head
        decoded_feat = self.decoder_gsfm(features)

        # 5. Dự đoán bản đồ xác suất (Probability Map) với kích thước gốc của ảnh đầu vào
        out = self.pred_head(decoded_feat)
        out = F.interpolate(out, size=original_size, mode="bilinear", align_corners=False)

        # 6. Áp dụng sigmoid để đưa ra xác suất trong khoảng [0, 1]
        return torch.sigmoid(out)
