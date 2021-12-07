import torch
import torch.nn as nn

anchors = [[10, 13, 16, 30, 33, 23],
           [30, 61, 62, 45, 59, 119],
           [116, 90, 156, 198, 373, 326]]


def check_anchor_order(m):
    # Check anchor order against stride order for YOLOv5 Head() module m,
    # and correct if necessary
    a = m.anchor_grid.prod(-1).view(-1)  # anchor area
    da = a[-1] - a[0]  # delta a
    ds = m.stride[-1] - m.stride[0]  # delta s
    if da.sign() != ds.sign():  # same order
        print('Reversing anchor order')
        m.anchor[:] = m.anchor.flip(0)
        m.anchor_grid[:] = m.anchor_grid.flip(0)


def pad(k, p=None):  # kernel, padding
    # Pad to 'same'
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


class DarkNet(nn.Module):
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
        b1 = self.b1(b0)  # 1-P1/2
        b2 = self.b2(b1)
        b3 = self.b3(b2)  # 3-P2/4
        b4 = self.b4(b3)
        b5 = self.b5(b4)  # 5-P3/8
        b6 = self.b6(b5)
        b7 = self.b7(b6)  # 7-P4/16
        b8 = self.b8(b7)
        b9 = self.b9(b8)  # 9-P5/32
        b10 = self.b10(b9)

        return b6, b8, b10

    @staticmethod
    def _make_layers(block, channels, num_blocks):
        layers = [block(channels, channels) for _ in range(num_blocks)]
        return nn.Sequential(*layers)


class FPN(nn.Module):
    def __init__(self):
        super(FPN, self).__init__()
        self.h11 = Bottleneck(1024, 1024, shortcut=False)  # 11
        self.h12 = Conv(1024, 512, 1, 1)  # 12
        self.h13 = Conv(512, 1024, 3, 1)  # 13
        self.h14 = Conv(1024, 512, 1, 1)  # 14
        self.h15 = Conv(512, 1024, 3, 1)  # 15 (P5/32-large)

        self.h16 = Conv(512, 256, 1, 1)  # 16
        self.h17 = nn.Upsample(None, scale_factor=2, mode='nearest')  # 17
        # self.h18: cat backbone P4 # 18
        self.h19 = Bottleneck(768, 512, shortcut=False)  # 19
        self.h20 = Bottleneck(512, 512, shortcut=False)  # 20
        self.h21 = Conv(512, 256, 1, 1)  # 21
        self.h22 = Conv(256, 512, 3, 1)  # 22 (P4/16-medium)

        self.h23 = Conv(256, 128, 1, 1)  # 23
        self.h24 = nn.Upsample(None, scale_factor=2, mode='nearest')  # 24
        # self.h25: cat backbone P3 # 25
        self.h26 = Bottleneck(384, 256, shortcut=False)  # 26
        self.h27 = Bottleneck(256, 256, shortcut=False)  # 27 (P3/8-small)

    def forward(self, x):
        p3, p4, p5 = x

        h11 = self.h11(p5)
        h12 = self.h12(h11)
        h13 = self.h13(h12)
        h14 = self.h14(h13)
        h15 = self.h15(h14)  # 15 (P5/32-large)

        h16 = self.h16(h14)
        h17 = self.h17(h16)
        h18 = torch.cat([h17, p4], dim=1)
        h19 = self.h19(h18)
        h20 = self.h20(h19)
        h21 = self.h21(h20)
        h22 = self.h22(h21)  # 22 (P4/16-medium)

        h23 = self.h23(h21)
        h24 = self.h24(h23)
        h25 = torch.cat([h24, p3], dim=1)
        h26 = self.h26(h25)
        h27 = self.h27(h26)  # 27 (P3/8-small)

        return h15, h22, h27


class Detect(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self.make_grid(nx, ny, i)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    def make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)], indexing='ij')

        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        anchor_grid = (self.anchors[i].clone() * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(
            (1, self.na, ny, nx, 2)).float()
        return grid, anchor_grid
