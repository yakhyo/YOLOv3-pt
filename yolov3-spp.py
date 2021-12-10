import torch
import torch.nn as nn

anchors = [[10, 13, 16, 30, 33, 23],  # P3/8
           [30, 61, 62, 45, 59, 119],  # P4/16
           [116, 90, 156, 198, 373, 326]]  # P5/32


# Pad to 'same'
def _pad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


# Standard convolution
class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, _pad(k, p), groups=g, bias=False)
        self.norm = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


# Residual bottleneck
class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # e-expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)
        self.conv1 = Conv(c1, c_, 1, 1)
        self.conv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))


# Spatial Pyramid Pooling (SPP) layer
class SPP(nn.Module):
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        c_ = c1 // 2
        self.conv1 = Conv(c1, c_, 1, 1)
        self.conv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)

        self.pool0 = nn.MaxPool2d(kernel_size=k[0], stride=1, padding=_pad(k[0]))
        self.pool1 = nn.MaxPool2d(kernel_size=k[1], stride=1, padding=_pad(k[1]))
        self.pool2 = nn.MaxPool2d(kernel_size=k[2], stride=1, padding=_pad(k[2]))

    def forward(self, x):
        x = self.conv1(x)
        pool0 = self.pool0(x)
        pool1 = self.pool1(x)
        pool2 = self.pool2(x)
        x = torch.cat([x, pool0, pool1, pool2], dim=1)

        return self.conv2(x)


# Spatial Pyramid Pooling - Fast (SPPF)
class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super(SPPF, self).__init__()
        c_ = c1 // 2
        self.conv1 = Conv(c1, c_, 1, 1)
        self.conv2 = Conv(4 * c_, c2, 1, 1)
        self.pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=pad(k))

    def forward(self, x):
        x = self.conv1(x)
        pool0 = self.pool(x)
        pool1 = self.pool(pool0)
        pool2 = self.pool(pool1)
        x = torch.cat([x, pool0, pool1, pool2], dim=1)

        return self.conv2(x)


# YOLOv3 SPP backbone
class DarkNet(nn.Module):
    def __init__(self, filters, block):
        super(DarkNet, self).__init__()
        self.b0 = Conv(3, 32, 3, 1)  # 0
        self.b1 = Conv(32, 64, 3, 2)  # 1-P1/2
        self.b2 = self._make_layers(Bottleneck, 64, num_blocks=1)  # 2
        self.b3 = Conv(64, 128, 3, 2)  # 3-P2/4
        self.b4 = self._make_layers(Bottleneck, 128, num_blocks=2)  # 4
        self.b5 = Conv(128, 256, 3, 2)  # 5-P3/8
        self.b6 = self._make_layers(Bottleneck, 256, num_blocks=8)  # 6
        self.b7 = Conv(256, 512, 3, 2)  # 7-P4/16
        self.b8 = self._make_layers(Bottleneck, 512, num_blocks=8)  # 8
        self.b9 = Conv(512, 1024, 3, 2)  # 9-P5/32
        self.b10 = self._make_layers(Bottleneck, 1024, num_blocks=4)  # 10

    def forward(self, x):
        b0 = self.b0(x)
        b1 = self.b1(b0)
        b2 = self.b2(b1)
        b3 = self.b3(b2)
        b4 = self.b4(b3)
        b5 = self.b5(b4)
        b6 = self.b6(b5)
        b7 = self.b7(b6)
        b8 = self.b8(b7)
        b9 = self.b9(b8)
        b10 = self.b10(b9)
        return b4, b8, b10


# YOLOv3 SPP head
class Head(nn.Module):
    def __init__(self, filters, blocks):
        super(Head, self).__init__()
        self.h11 = Bottleneck(1024, 1024, shortcut=False)  # 11
        self.h12 = SPP(1024, 512, k=(5, 9, 13))  # 12
        self.h13 = Conv(512, 1024, 3, 1)  # 13
        self.h14 = Conv(1024, 512, 1, 1)  # 14
        self.h15 = Conv(512, 1024, 3, 1)  # 15 (P5/32-large)

        self.h16 = Conv(512, 256, 1, 1)  # 16
        self.h17 = nn.Upsample(None, scale_factor=2, mode='nearest')  # 17
        # self.h18 cat backbone P4 # 18
        self.h19 = Bottleneck(256, 512, shortcut=False)  # 19
        self.h20 = Bottleneck(512, 512, shortcut=False)  # 20
        self.h21 = Conv(512, 256, 1, 1)  # 21
        self.h22 = Conv(256, 512, 3, 1)  # 22 (P4/16-medium)

        self.h23 = Conv(256, 128, 1, 1)  # 23
        self.h24 = nn.Upsample(None, scale_factor=2, mode='nearest')  # 24
        # self.h25 cat backbone P3 # 25
        self.h26 = Bottleneck(128, 256, shortcut=False)  # 26
        self.h27 = nn.Sequential(Bottleneck(256, 256, shortcut=False),  # 27 (P3/8-small)
                                 Bottleneck(256, 256, shortcut=False))

    def forward(self, x):
        p3, p4, p5 = x
        h11 = self.h11(p5)
        h12 = self.h12(h11)