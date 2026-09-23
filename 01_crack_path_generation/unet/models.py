import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, p_drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p_drop),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_ch=2, base=32, p_drop=0.15):
        super().__init__()
        self.d1 = DoubleConv(in_ch, base, p_drop)
        self.p1 = nn.MaxPool2d(2)
        self.d2 = DoubleConv(base, base*2, p_drop)
        self.p2 = nn.MaxPool2d(2)
        self.d3 = DoubleConv(base*2, base*4, p_drop)
        self.p3 = nn.MaxPool2d(2)
        self.d4 = DoubleConv(base*4, base*8, p_drop)

        self.u3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.up3 = DoubleConv(base*8, base*4, p_drop)
        self.u2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.up2 = DoubleConv(base*4, base*2, p_drop)
        self.u1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.up1 = DoubleConv(base*2, base, p_drop)

        self.out = nn.Conv2d(base, 1, 1)  # crack logits

    def forward(self, x):
        c1 = self.d1(x)
        c2 = self.d2(self.p1(c1))
        c3 = self.d3(self.p2(c2))
        c4 = self.d4(self.p3(c3))

        x = self.u3(c4)
        x = torch.cat([x, c3], dim=1)
        x = self.up3(x)

        x = self.u2(x)
        x = torch.cat([x, c2], dim=1)
        x = self.up2(x)

        x = self.u1(x)
        x = torch.cat([x, c1], dim=1)
        x = self.up1(x)

        return self.out(x)
