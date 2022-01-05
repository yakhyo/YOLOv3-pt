import torch
import torch.nn as nn


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
        self.pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=_pad(k))

    def forward(self, x):
        x = self.conv1(x)
        pool0 = self.pool(x)
        pool1 = self.pool(pool0)
        pool2 = self.pool(pool1)
        x = torch.cat([x, pool0, pool1, pool2], dim=1)

        return self.conv2(x)


# Concatenating list of layers
class Concat(nn.Module):

    def __init__(self, d=1):
        super().__init__()
        self.d = d

    def __call__(self, x):
        return torch.cat(x, self.d)
