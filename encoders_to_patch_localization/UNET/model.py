import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    # Changed default out_channels to 19 for BDD100K
    def __init__(self, in_channels=3, out_channels=19): 
        super(UNet, self).__init__()
        
        self.down1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        self.bottleneck = DoubleConv(256, 512)
        
        self.upConv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up1 = DoubleConv(512, 256) 
        
        self.upConv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up2 = DoubleConv(256, 128)
        
        self.upConv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up3 = DoubleConv(128, 64)
        
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.pool1(x1)
        x3 = self.down2(x2)
        x4 = self.pool2(x3)
        x5 = self.down3(x4)
        x6 = self.pool3(x5)
        
        b = self.bottleneck(x6)
        
        d1 = self.upConv1(b)
        d2 = self.up1(torch.cat((x5, d1), dim=1))
        
        d3 = self.upConv2(d2)
        d4 = self.up2(torch.cat((x3, d3), dim=1))
        
        d5 = self.upConv3(d4)
        d6 = self.up3(torch.cat((x1, d5), dim=1))
        
        return self.outc(d6)