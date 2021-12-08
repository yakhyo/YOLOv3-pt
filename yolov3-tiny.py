import torch
import torch.nn as nn

anchors = [[10, 14, 23, 27, 37, 58],
           [81, 82, 135, 169, 344, 319]]


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


class DarkNet(nn.Module):
    # YOLOv3 tiny backbone
    def __init__(self):
        super(DarkNet, self).__init__()
        self.b0 = Conv(3, 16, 3, 1)  # 0
        self.b1 = nn.MaxPool2d(2, 2)  # 1-P1/2
        self.b2 = Conv(16, 32, 3, 1)  # 2
        self.b3 = nn.MaxPool2d(2, 2)  # 3-P2/4
        self.b4 = Conv(32, 64, 3, 1)  # 4
        self.b5 = nn.MaxPool2d(2, 2)  # 5-P3/8
        self.b6 = Conv(64, 128, 3, 1)  # 6
        self.b7 = nn.MaxPool2d(2, 2)  # 7-P4/16
        self.b8 = Conv(128, 256, 3, 1)  # 8
        self.b9 = nn.MaxPool2d(2, 2)  # 9-P5/32
        self.b10 = Conv(256, 512, 3, 1)  # 10
        self.b11 = nn.ZeroPad2d((0, 1, 0, 1))  # 11
        self.b12 = nn.MaxPool2d(2, 1)  # 12

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
        b11 = self.b11(b10)
        b12 = self.b12(b11)

        return b8, b12


class Head(nn.Module):
    # YOLOv3 tiny head
    def __init__(self):
        super(Head, self).__init__()
        self.h13 = Conv(512, 1024, 3, 1)  # 13
        self.h14 = Conv(1024, 256, 1, 1)  # 14
        self.h15 = Conv(256, 512, 3, 1)  # 15 (P5/32-large)

        self.h16 = Conv(256, 128, 1, 1)  # 16
        self.h17 = nn.Upsample(None, scale_factor=2, mode='nearest')  # 17
        # self.h18 Concat backbone P4 # 18
        self.h19 = Conv(384, 256, 3, 1)  # 19 (P4/16-medium)

    def forward(self, x):
        p4, p5 = x
        h13 = self.h13(p5)
        h14 = self.h14(h13)
        h15 = self.h15(h14)  # 15 (P5/32-large)

        h16 = self.h16(h14)
        h17 = self.h17(h16)
        h18 = torch.cat([h17, p4], dim=1)
        h19 = self.h19(h18)  # 19 (P4/16-medium)

        return h15, h19


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


class YOLOv3t(nn.Module):
    def __init__(self, num_classes):
        super(YOLOv3t, self).__init__()


if __name__ == '__main__':
    a = torch.randn(1, 3, 256, 256)
    net = DarkNet()
    head = Head()
    p4, p5 = net(a)

    print(p4.shape, p5.shape)

    p4, p5 = head((p4, p5))
    print(p4.shape, p5.shape)
