import torch
import torch.nn as nn

base = {
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
}

# CONV_DEFS_16 = [
#     Conv(stride=1, depth=64),
#     Conv(stride=1, depth=64),
#     'M',
#     Conv(stride=1, depth=128),
#     Conv(stride=1, depth=128),
#     'M'
#     Conv(stride=1, depth=256),
#     Conv(stride=1, depth=256),
#     Conv(stride=1, depth=256),
#     'M'
#     Conv(stride=1, depth=512),
#     Conv(stride=1, depth=512),
#     Conv(stride=1, depth=512),
#     'M'
#     Conv(stride=1, depth=512),
#     Conv(stride=1, depth=512),
#     Conv(stride=1, depth=512),
# ]

# Conv = namedtuple('Conv', ['stride', 'depth'])

# class _conv_bn(nn.Module):
#     def __init__(self, inp, oup, stride):
#         super(_conv_bn, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
#             nn.BatchNorm2d(oup),
#             nn.ReLU(inplace=True),
#         )
#         self.depth = oup

#     def forward(self, x):
#         return self.conv(x)


def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    layers += [
        nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
        nn.ReLU(inplace=True),
        nn.Conv2d(1024, 1024, kernel_size=1),
        nn.ReLU(inplace=True)]
    return layers

def vgg16():
    return vgg(base['vgg16'], 3)
vgg16.name='vgg16'