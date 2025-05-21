import torch
import torch.nn as nn
import torch.nn.functional as F

class CSPUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CSPUp, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        self.compress = nn.Conv2d(in_channels, out_channels // 2, kernel_size=1)
        
        # Upsampling
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.2)
    
    def forward(self, x):
        x_main = self.upsample(x)
        x_main = self.leakyrelu(self.bn1(self.conv1(x_main)))
        x_main = self.leakyrelu(self.bn2(self.conv2(x_main)))
        
        x_comp = self.compress(x)
        x_comp = self.upsample(x_comp)
        
        out = torch.cat([x_main, x_comp], dim=1)
        return out