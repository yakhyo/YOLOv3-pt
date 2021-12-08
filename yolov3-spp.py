import torch
import torch.nn as nn

anchors = [[10, 13, 16, 30, 33, 23],  # P3/8
           [30, 61, 62, 45, 59, 119],  # P4/16
           [116, 90, 156, 198, 373, 326]]  # P5/32


def pad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, pad(k, p), groups=g, bias=False)
        self.norm = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # e-expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)
        self.conv1 = Conv(c1, c_, 1, 1)
        self.conv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))


class SPP(nn.Module):
    # Spatial Pyramid Pooling (SPP) layer
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        c_ = c1 // 2
        self.conv1 = Conv(c1, c_, 1, 1)
        self.conv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)

        self.pool0 = nn.MaxPool2d(k[0], 1, pad(k[0]))
        self.pool1 = nn.MaxPool2d(k[1], 1, pad(k[1]))
        self.pool2 = nn.MaxPool2d(k[2], 1, pad(k[2]))

    def forward(self, x):
        x = self.conv1(x)
        pool0 = self.pool0(x)
        pool1 = self.pool1(x)
        pool2 = self.pool2(x)
        x = torch.cat([x, pool0, pool1, pool2], 1)

        return self.conv2(x)


class DarkNet(nn.Module):
    # YOLOv3 SPP backbone
    def __init__(self):
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


class Head(nn.Module):
    # YOLOv3 SPP head
    def __init__(self):
        super(Head, self).__init__()
        self.h11 =

    @staticmethod
    def _make_layers(block, channels, num_blocks):
        layers = [block(channels, channels) for _ range(num_blocks)]
        return nn.Sequential(*layers)
