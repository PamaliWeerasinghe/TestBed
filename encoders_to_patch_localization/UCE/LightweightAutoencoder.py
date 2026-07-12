import torch.nn as nn

class LightweightAutoencoder(nn.Module):
    def __init__(self):
        super(LightweightAutoencoder, self).__init__()
        
        # Encoder: Compresses 128x128 -> 64x64 -> 32x32 -> 16x16
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # 16 x 64 x 64
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 32 x 32 x 32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 64 x 16 x 16
            nn.LeakyReLU(0.2) # Avoid dead neurons in the bottleneck
        )
        
        # Decoder: Smoothly reconstructs 16x16 -> 32x32 -> 64x64 -> 128x128
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1), 
            nn.Sigmoid() 
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))