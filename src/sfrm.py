import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

class SpatialFrequencyModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 0. Ghép 2 đầu vào (kết quả bước trước + feature nông hơn) thành 1
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        # 1. Luồng không gian (Spatial Stream): Dùng Conv2d thông thường
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # 2. Luồng tần số (Frequency Stream): Xử lý biên độ tần số qua phép biến đổi FFT
        #    Đầu ra 2*channels = phần thực + phần ảo, để ghép lại thành số phức rồi IFFT
        self.freq_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels * 2, kernel_size=1),
            nn.BatchNorm2d(channels * 2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_deep, x_shallow):
        # x_deep:    kết quả đã trộn của bước trước (nhỏ hơn)   [Batch, Channels, h, w]
        # x_shallow: feature nông hơn, chi tiết hơn (lớn hơn)    [Batch, Channels, H, W]
        target_size = x_shallow.shape[2:]

        # Phóng to x_deep bằng x_shallow, ghép lại rồi conv
        x_deep = F.interpolate(x_deep, size=target_size, mode='bilinear', align_corners=False)
        x = self.fuse(torch.cat([x_deep, x_shallow], dim=1))

        # Xử lý luồng không gian
        spatial_out = self.spatial_conv(x)

        # Xử lý luồng tần số: FFT -> conv trong miền tần số -> IFFT quay về miền không gian
        fft_feat = torch.fft.rfft2(x.float(), norm='ortho')
        # Chuyển đổi kết quả FFT sang dạng thực và ảo để đưa vào Conv2d
        fft_cat = torch.cat([fft_feat.real, fft_feat.imag], dim=1)
        real, imag = self.freq_conv(fft_cat).chunk(2, dim=1)
        # .float(): khi train bằng fp16 (AMP), IFFT cần số thực 32-bit
        freq_feat = torch.fft.irfft2(torch.complex(real.float(), imag.float()), s=target_size, norm='ortho')

        # Kết hợp cả hai luồng lại với nhau (+ x để giữ thông tin gốc)
        return spatial_out + freq_feat + x