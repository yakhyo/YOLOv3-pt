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
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # e-expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)
        self.conv1 = Conv(c1, c_, 1, 1)
        self.conv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))


# YOLOv3 backbone
class DarkNet(nn.Module):
    def __init__(self, filters, blocks):
        super(DarkNet, self).__init__()
        self.b0 = Conv(filters[0], filters[1], 3, 1)  # 0
        self.b1 = Conv(filters[1], filters[2], 3, 2)  # 1-P1/2
        self.b2 = self._make_layers(Bottleneck, filters[2], num_blocks=blocks[0])  # 2
        self.b3 = Conv(filters[2], filters[3], 3, 2)  # 3-P2/4
        self.b4 = self._make_layers(Bottleneck, filters[3], num_blocks=blocks[1])  # 4
        self.b5 = Conv(filters[3], filters[4], 3, 2)  # 5-P3/8
        self.b6 = self._make_layers(Bottleneck, filters[4], num_blocks=blocks[3])  # 6
        self.b7 = Conv(filters[4], filters[5], 3, 2)  # 7-P4/16
        self.b8 = self._make_layers(Bottleneck, filters[5], num_blocks=blocks[3])  # 8
        self.b9 = Conv(filters[5], filters[6], 3, 2)  # 9-P5/32
        self.b10 = self._make_layers(Bottleneck, filters[6], num_blocks=blocks[2])  # 10

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


# YOLOv3 head - FPN (Feature Pyramid Network)
class Head(nn.Module):
    def __init__(self):
        super(Head, self).__init__()
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
        self.h27 = nn.Sequential(Bottleneck(256, 256, shortcut=False),  # 27 (P3/8-small)
                                 Bottleneck(256, 256, shortcut=False))

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


# YOLOv3 detection head
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
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

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
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for  on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)


class YOLOv3(nn.Module):
    def __init__(self, num_classes):
        super(YOLOv3, self).__init__()
        blocks = [1, 2, 4, 8]
        filters = [3, 32, 64, 128, 256, 512, 1024]
        self.backbone = DarkNet(filters, blocks)


if __name__ == '__main__':
    blocks = [1, 2, 4, 8]
    filters = [3, 32, 64, 128, 256, 512, 1024]
    backbone = DarkNet(filters, blocks)
    a = torch.randn(1, 3, 640, 640)
    p3, p4, p5 = backbone(a)
    head = Head()

    h3, h4, h5 = head((p3, p4, p5))
    print("Backbone")
    print(p3.shape, p4.shape, p5.shape)
    print("Head")
    print(h3.shape, h4.shape, h5.shape)
