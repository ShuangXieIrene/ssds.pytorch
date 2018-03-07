import torch
import torch.nn as nn

from collections import namedtuple
import functools

BasicBlock = namedtuple('BasicBlock', ['stride', 'depth', 'num', 't'])
Bottleneck = namedtuple('Bottleneck', ['stride', 'depth', 'num', 't']) # t is the expension factor

V18_CONV_DEFS = [
    BasicBlock(stride=1, depth=64, num=2, t=1),
    BasicBlock(stride=2, depth=128, num=2, t=1),
    BasicBlock(stride=2, depth=256, num=2, t=1),
    # BasicBlock(stride=2, depth=512, num=2, t=1),
]

V34_CONV_DEFS = [
    BasicBlock(stride=1, depth=64, num=3, t=1),
    BasicBlock(stride=2, depth=128, num=4, t=1),
    BasicBlock(stride=2, depth=256, num=6, t=1),
    # BasicBlock(stride=2, depth=512, num=3, t=1),
]

V50_CONV_DEFS = [
    Bottleneck(stride=1, depth=64, num=3, t=4),
    Bottleneck(stride=2, depth=128, num=4, t=4),
    Bottleneck(stride=2, depth=256, num=6, t=4),
    # Bottleneck(stride=2, depth=512, num=3, t=4),
]

V101_CONV_DEFS = [
    Bottleneck(stride=1, depth=64, num=3, t=4),
    Bottleneck(stride=2, depth=128, num=4, t=4),
    Bottleneck(stride=2, depth=256, num=23, t=4),
    # Bottleneck(stride=2, depth=512, num=3, t=4),
]

class _basicblock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, expansion=1, downsample=None):
        super(_basicblock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes * expansion, kernel_size=3, stride=1,
                     padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes * expansion)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class _bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, expansion=4, downsample=None):
        super(_bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def resnet(conv_defs, depth_multiplier=1.0, min_depth=8):
    depth = lambda d: max(int(d * depth_multiplier), min_depth)
    layers = [
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        ]
    in_channels = 64
    for conv_def in conv_defs:
        if conv_def.stride != 1 or in_channels != depth(conv_def.depth * conv_def.t):
            _downsample = nn.Sequential(
                nn.Conv2d(in_channels, depth(conv_def.depth * conv_def.t),
                          kernel_size=1, stride=conv_def.stride, bias=False),
                nn.BatchNorm2d(depth(conv_def.depth * conv_def.t)),
            )
        if isinstance(conv_def, BasicBlock):
          for n in range(conv_def.num):
            (stride, downsample) = (conv_def.stride, _downsample) if n == 0 else (1, None)
            layers += [_basicblock(in_channels, depth(conv_def.depth), stride, conv_def.t, downsample)]
            in_channels = depth(conv_def.depth * conv_def.t)
        elif isinstance(conv_def, Bottleneck):
          for n in range(conv_def.num):
            (stride, downsample) = (conv_def.stride, _downsample) if n == 0 else (1, None)
            layers += [_bottleneck(in_channels, depth(conv_def.depth), stride, conv_def.t, downsample)]
            in_channels = depth(conv_def.depth * conv_def.t)
    return layers

def wrapped_partial(func, *args, **kwargs):
    partial_func = functools.partial(func, *args, **kwargs)
    functools.update_wrapper(partial_func, func)
    return partial_func

resnet_18 = wrapped_partial(resnet, conv_defs=V18_CONV_DEFS, depth_multiplier=1.0)
resnet_34 = wrapped_partial(resnet, conv_defs=V34_CONV_DEFS, depth_multiplier=1.0)

resnet_50 = wrapped_partial(resnet, conv_defs=V50_CONV_DEFS, depth_multiplier=1.0)
resnet_101 = wrapped_partial(resnet, conv_defs=V101_CONV_DEFS, depth_multiplier=1.0)
