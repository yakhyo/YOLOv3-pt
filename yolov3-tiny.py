import torch
import torch.nn as nn
import math

anchors = [[10, 14, 23, 27, 37, 58],
           [81, 82, 135, 169, 344, 319]]


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


# YOLOv3 tiny backbone
class DarkNet(nn.Module):
    def __init__(self, filters):
        super(DarkNet, self).__init__()
        self.b0 = Conv(filters[0], filters[1], 3, 1)  # 0
        self.b1 = nn.MaxPool2d(2, 2)  # 1-P1/2
        self.b2 = Conv(filters[1], filters[2], 3, 1)  # 2
        self.b3 = nn.MaxPool2d(2, 2)  # 3-P2/4
        self.b4 = Conv(filters[2], filters[3], 3, 1)  # 4
        self.b5 = nn.MaxPool2d(2, 2)  # 5-P3/8
        self.b6 = Conv(filters[3], filters[4], 3, 1)  # 6
        self.b7 = nn.MaxPool2d(2, 2)  # 7-P4/16
        self.b8 = Conv(filters[4], filters[5], 3, 1)  # 8
        self.b9 = nn.MaxPool2d(2, 2)  # 9-P5/32
        self.b10 = Conv(filters[5], filters[6], 3, 1)  # 10
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


# YOLOv3 tiny head
class Head(nn.Module):
    def __init__(self, filters):
        super(Head, self).__init__()
        self.h13 = Conv(filters[6], filters[7], 3, 1)  # 13
        self.h14 = Conv(filters[7], filters[5], 1, 1)  # 14
        self.h15 = Conv(filters[5], filters[6], 3, 1)  # 15 (P5/32-large)

        self.h16 = Conv(filters[5], filters[4], 1, 1)  # 16
        self.h17 = nn.Upsample(None, scale_factor=2, mode='nearest')  # 17
        # self.h18 Concat backbone P4 # 18
        self.h19 = Conv(filters[4] + filters[5], filters[5], 3, 1)  # 19 (P4/16-medium)

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


# YOLOv3 detection head
class Detect(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = torch.tensor([torch.zeros(1)] * self.nl)  # init anchor grid
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
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)], indexing='ij')
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        anchor_grid = (self.anchors[i].clone() * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(
            (1, self.na, ny, nx, 2)).float()
        return grid, anchor_grid


# YOLOv3 Tiny model
class YOLOv3(nn.Module):
    def __init__(self):
        super(YOLOv3, self).__init__()
        filters = [3, 16, 32, 64, 128, 256, 512, 1024]
        self.backbone = DarkNet(filters)
        self.head = Head(filters)
        self.detect = Detect(anchors=anchors, ch=(512, 256))
        img = torch.zeros(1, 3, 256, 256)
        self.detect.stride = torch.tensor([256 / x.shape[-2] for x in self.forward(img)])
        self.detect.anchors /= self.detect.stride.view(-1, 1, 1)
        self._check_anchor_order(self.detect)
        self._initialize_biases()

    def forward(self, x):
        p4, p5 = self.backbone(x)
        p4, p5 = self.head([p4, p5])
        return self.detect([p4, p5])

    def _initialize_biases(self):  # initialize biases into Detect()
        for m, s in zip(self.detect.m, self.detect.stride):  # from
            b = m.bias.view(self.detect.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (self.detect.nc - 0.999999))
            m.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    # Reverse anchor order
    @staticmethod
    def _check_anchor_order(m):
        a = m.anchor_grid.prod(-1).view(-1)  # anchor area
        da = a[-1] - a[0]  # delta a
        ds = m.stride[-1] - m.stride[0]  # delta s
        if da.sign() != ds.sign():  # same order
            m.anchors[:] = m.anchors.flip(0)
            m.anchor_grid[:] = m.anchor_grid.flip(0)


if __name__ == '__main__':
    net = YOLOv3()
    img = torch.randn(1, 3, 640, 640)
    res = net(img)
    print("Num. of parameters: {}".format(sum(p.numel() for p in net.parameters() if p.requires_grad)))
