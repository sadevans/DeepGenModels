import torch.nn as nn
from .cspup import CSPUp

class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super(Generator, self).__init__()
        
        # Начальный fully-connected слой
        self.fc = nn.Linear(z_dim, 4*4*1024)
        self.bn0 = nn.BatchNorm1d(4*4*1024)
        self.leakyrelu = nn.LeakyReLU(0.2)
        
        # Блоки CSPUp
        self.cspup1 = CSPUp(1024, 512)
        self.cspup2 = CSPUp(1024, 512)  # Учитываем конкатенацию каналов
        
        # Дополнительные слои upsampling
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )
        
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        
        # Финальный слой
        self.final = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )
    
    def forward(self, z):
        # Преобразование входного шума
        x = self.fc(z)
        x = self.bn0(x)
        x = self.leakyrelu(x)
        x = x.view(-1, 1024, 4, 4)
        
        # Применение CSPUp блоков
        x = self.cspup1(x)
        x = self.cspup2(x)
        
        # Дополнительные upsampling слои
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        
        # Финальное преобразование
        x = self.final(x)
        return x