import torch
import torch.nn as nn
import torch.nn.functional as F

from src.sfrm import SpatialFrequencyModule

class GSFM(nn.Module):
    def __init__(self, channels=256):
        super().__init__()
        # 3 khối SFRM cho 3 bước trộn (công thức 2 của bài báo)
        self.sfrm_43 = SpatialFrequencyModule(channels)   # bước 1: trộn F4 với F3
        self.sfrm_32 = SpatialFrequencyModule(channels)   # bước 2: trộn kết quả với F2
        self.sfrm_21 = SpatialFrequencyModule(channels)   # bước 3: trộn kết quả với F1

    def forward(self, features):
        # features là danh sách các tầng đặc trưng từ Decoder [F1, F2, F3, F4]
        # F4 nhỏ nhất (biết "vật nằm đâu") ... F1 lớn nhất (biết "biên ở đâu")
        F1, F2, F3, F4 = features

        # Trộn NỐI TẦNG: kết quả bước trước là đầu vào của bước sau
        #   Fo = SFRM(SFRM(SFRM(F4, F3), F2), F1)
        x = self.sfrm_43(F4, F3)
        x = self.sfrm_32(x, F2)
        x = self.sfrm_21(x, F1)
        return x