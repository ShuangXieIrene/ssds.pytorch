import torch
import torch.nn as nn


class SepConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, expand_ratio=1):
        padding = (kernel_size - 1) // 2
        super(SepConvBNReLU, self).__init__(
            # dw
            nn.Conv2d(
                in_planes,
                in_planes,
                kernel_size,
                stride,
                padding,
                groups=in_planes,
                bias=False,
            ),
            nn.BatchNorm2d(in_planes),
            nn.ReLU(inplace=True),
            # pw
            nn.Conv2d(in_planes, out_planes, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
        )


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(
                in_planes, out_planes, kernel_size, stride, padding=padding, bias=False
            ),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
        )


class ConvBNReLUx2(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLUx2, self).__init__(
            nn.Conv2d(in_planes, out_planes // 2, 1, bias=False),
            nn.BatchNorm2d(out_planes // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_planes // 2,
                out_planes,
                kernel_size,
                stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
        )
