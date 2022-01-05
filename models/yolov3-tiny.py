"""
@author: Yakhyokhuja Valikhujaev <yakhyo9696@gmail.com>
"""

import math
import torch
import torch.nn as nn
from common import Conv, Concat, DETECT

depth_multiple = 1.0  # model depth multiple
width_multiple = 1.0  # layer channel multiple

anchors = [[10, 14, 23, 27, 37, 58],
           [81, 82, 135, 169, 344, 319]]


# YOLOv3 Tiny Backbone
class BACKBONE(nn.Module):

    def __init__(self, filters):
        super(BACKBONE, self).__init__()
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


# YOLOv3 Tiny Head
class HEAD(nn.Module):

    def __init__(self, filters):
        super(HEAD, self).__init__()
        self.h13 = Conv(filters[6], filters[7], 3, 1)  # 13
        self.h14 = Conv(filters[7], filters[5], 1, 1)  # 14
        self.h15 = Conv(filters[5], filters[6], 3, 1)  # 15 (P5/32-large)

        self.h16 = Conv(filters[5], filters[4], 1, 1)  # 16
        self.h17 = nn.Upsample(None, scale_factor=2, mode='nearest')  # 17
        self.h18 = Concat()  # Concat backbone P4 # 18
        self.h19 = Conv(filters[4] + filters[5], filters[5], 3, 1)  # 19 (P4/16-medium)

    def forward(self, x):
        p4, p5 = x
        h13 = self.h13(p5)
        h14 = self.h14(h13)
        h15 = self.h15(h14)  # 15 (P5/32-large)

        h16 = self.h16(h14)
        h17 = self.h17(h16)
        h18 = self.h18([h17, p4])
        h19 = self.h19(h18)  # 19 (P4/16-medium)

        return h19, h15


# YOLOv3 Tiny Model
class YOLOv3(nn.Module):

    def __init__(self, anchors):
        super(YOLOv3, self).__init__()

        filters = [3, 16, 32, 64, 128, 256, 512, 1024]

        filters = [3, *[self._make_divisible(c * width_multiple, 8) for c in filters[1:]]]

        self.backbone = BACKBONE(filters)
        self.head = HEAD(filters)
        self.detect = DETECT(anchors=anchors, ch=(filters[5], filters[6]))

        dummy_img = torch.zeros(1, 3, 256, 256)
        self.detect.stride = torch.tensor([256 / x.shape[-2] for x in self.forward(dummy_img)])
        self.detect.anchors /= self.detect.stride.view(-1, 1, 1)
        self._check_anchor_order(self.detect)
        self._initialize_biases(self.detect)

    def forward(self, x):
        p4, p5 = self.backbone(x)
        p4, p5 = self.head([p4, p5])
        return self.detect([p4, p5])

    @staticmethod
    def _make_divisible(x, divisor):
        # Returns nearest x divisible by divisor
        if isinstance(divisor, torch.Tensor):
            divisor = int(divisor.max())  # to int
        return math.ceil(x / divisor) * divisor

    @staticmethod
    def _initialize_biases(detect):
        det = detect
        for layer, stride in zip(det.m, det.stride):
            b = layer.bias.view(det.na, -1)
            b.data[:, 4] += math.log(8 / (640 / stride) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (det.nc - 0.999999))
            layer.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    @staticmethod
    def _check_anchor_order(det):
        a = det.anchors.prod(-1).view(-1)  # anchor area
        da = a[-1] - a[0]  # delta a
        ds = det.stride[-1] - det.stride[0]  # delta s
        if da.sign() != ds.sign():  # same order
            det.anchors[:] = det.anchors.flip(0)


if __name__ == '__main__':
    net = YOLOv3(anchors=anchors)
    net.eval()

    img = torch.randn(1, 3, 640, 640)
    predictions, (p4, p5) = net(img)

    print(f'P4.size(): {p4.size()}, \nP5.size(): {p5.size()}')
    print("Number of parameters: {:.2f}M".format(sum(p.numel() for p in net.parameters() if p.requires_grad) / 1e6))
