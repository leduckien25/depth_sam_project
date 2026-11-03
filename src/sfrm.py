import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

class SpatialFrequencyModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 1. Luồng không gian (Spatial Stream): Dùng Conv2d thông thường
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # 2. Luồng tần số (Frequency Stream): Xử lý biên độ tần số qua phép biến đổi FFT
        self.freq_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x shape: [Batch, Channels, Height, Width]
        target_size = x.shape[2:] # Lấy kích thước gốc (Height, Width) của x
        
        # Xử lý luồng không gian
        spatial_out = self.spatial_conv(x)
        
        # Xử lý luồng tần số bằng biến đổi Fourier 2D (FFT)
        fft_feat = torch.fft.rfft2(x, norm='ortho')
        # Chuyển đổi kết quả FFT sang dạng thực và ảo để đưa vào Conv2d
        fft_cat = torch.cat([fft_feat.real, fft_feat.imag], dim=1)
        freq_feat = self.freq_conv(fft_cat)
        
        # Ép kích thước của freq_feat về lại đúng bằng target_size để khớp hoàn toàn với spatial_out
        freq_feat = F.interpolate(freq_feat, size=target_size, mode='bilinear', align_corners=False)
        
        # Kết hợp cả hai luồng lại với nhau
        return spatial_out + freq_feat