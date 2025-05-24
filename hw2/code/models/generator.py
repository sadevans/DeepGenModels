import torch.nn as nn
from .cspup import CSPUp

class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super(Generator, self).__init__()
        # Последовательные CSPup-блоки
        self.cspup1 = CSPUp(z_dim)   # 4x4 → 8x8
        self.cspup2 = CSPUp(z_dim//2)    # 8x8 → 16x16
        self.cspup3 = CSPUp(z_dim//4)    # 16x16 → 32x32
        self.cspup4 = CSPUp(z_dim//8)     # 32x32 → 64x64

        # Финальный deconv слой
        self.final = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, z):
        x = self.cspup1(z)
        x = self.cspup2(x)
        x = self.cspup3(x)
        x = self.cspup4(x)

        x = self.final(x)
        return x
