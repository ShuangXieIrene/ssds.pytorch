import torch
import torch.nn as nn

from collections import namedtuple
import functools

Conv = namedtuple('Conv', ['stride', 'depth'])
ConvBlock = namedtuple('ConvBlock', ['stride', 'depth', 'num', 't']) # t is the expension factor
ResidualBlock = namedtuple('ResidualBlock', ['stride', 'depth', 'num', 't']) # t is the expension factor


CONV_DEFS_19 = [
    Conv(stride=1, depth=32),
    'M',
    Conv(stride=1, depth=64),
    'M',
    ConvBlock(stride=1, depth=128, num=2, t=0.5),
    'M',
    ConvBlock(stride=1, depth=256, num=2, t=0.5),
    'M',
    ConvBlock(stride=1, depth=512, num=3, t=0.5),
    'M',
    ConvBlock(stride=1, depth=1024, num=3, t=0.5),
]

CONV_DEFS_53 = [
    Conv(stride=1, depth=32),
    ResidualBlock(stride=2, depth=64, num=2, t=0.5),
    ResidualBlock(stride=2, depth=128, num=3, t=0.5),
    ResidualBlock(stride=2, depth=256, num=9, t=0.5),
    ResidualBlock(stride=2, depth=512, num=9, t=0.5),
    ResidualBlock(stride=2, depth=1024, num=5, t=0.5),
]

class _conv_bn(nn.Module):
    def __init__(self, inp, oup, stride):
        super(_conv_bn, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            nn.BatchNorm2d(oup),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.depth = oup

    def forward(self, x):
        return self.conv(x)

class _conv_block(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio=0.5):
        super(_conv_block, self).__init__()
        if stride == 1 and inp == oup:
            depth = int(oup*expand_ratio)
            self.conv = nn.Sequential(
                nn.Conv2d(inp, depth, 1, 1, bias=False),
                nn.BatchNorm2d(depth),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(depth, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.LeakyReLU(0.1, inplace=True),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.LeakyReLU(0.1, inplace=True),
            )
        self.depth = oup

    def forward(self, x):
        return self.conv(x)


class _residual_block(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio=0.5):
        super(_residual_block, self).__init__()
        self.use_res_connect = stride == 1 and inp == oup
        if self.use_res_connect:
            depth = int(oup*expand_ratio)
            self.conv = nn.Sequential(
                nn.Conv2d(inp, depth, 1, 1, bias=False),
                nn.BatchNorm2d(depth),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(depth, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.LeakyReLU(0.1, inplace=True),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.LeakyReLU(0.1, inplace=True),
            )
        self.depth = oup

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


def darknet(conv_defs, depth_multiplier=1.0, min_depth=8):
    depth = lambda d: max(int(d * depth_multiplier), min_depth)
    layers = []
    in_channels = 3
    for conv_def in conv_defs:
        if isinstance(conv_def, Conv):
            layers += [_conv_bn(in_channels, depth(conv_def.depth), conv_def.stride)]
            in_channels = depth(conv_def.depth)
        elif isinstance(conv_def, ConvBlock):
          for n in range(conv_def.num):
            stride = conv_def.stride if n == 0 else 1
            layers += [_conv_block(in_channels, depth(conv_def.depth), stride, conv_def.t)]
            in_channels = depth(conv_def.depth)
        elif isinstance(conv_def, ResidualBlock):
          for n in range(conv_def.num):
            stride = conv_def.stride if n == 0 else 1
            layers += [_residual_block(in_channels, depth(conv_def.depth), stride, conv_def.t)]
            in_channels = depth(conv_def.depth)
        elif conv_def == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
    return layers

def wrapped_partial(func, *args, **kwargs):
    partial_func = functools.partial(func, *args, **kwargs)
    functools.update_wrapper(partial_func, func)
    return partial_func

darknet_19 = wrapped_partial(darknet, conv_defs=CONV_DEFS_19, depth_multiplier=1.0)
darknet_53 = wrapped_partial(darknet, conv_defs=CONV_DEFS_53, depth_multiplier=1.0)
