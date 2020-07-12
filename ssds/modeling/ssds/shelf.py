import torch
import torch.nn as nn
from collections import OrderedDict

from .ssdsbase import SSDSBase
from ssds.modeling.layers.layers_parser import parse_feature_layer
from ssds.modeling.layers.basic_layers import ConvBNReLU


class SharedBlock(nn.Module):
    """ The conv params in this block is shared
    """

    def __init__(self, planes):
        super(SharedBlock, self).__init__()

        self.planes = planes
        self.conv1 = nn.Conv2d(
            self.planes, self.planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(p=0.25)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.drop(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = out + x
        return self.relu2(out)


class ShelfPyramid(nn.Module):
    def __init__(self, settings, conv=nn.ConvTranspose2d, block=SharedBlock):
        super().__init__()

        # "output_padding":1 is not work for tensorrt
        extra_args = {"padding": 1, "bias": True} if conv == nn.ConvTranspose2d else {}
        for i, depth in enumerate(settings):
            if i == 0:
                self.add_module("block{}".format(i), block(depth))
            else:
                self.add_module("block{}".format(i), block(depth))
                self.add_module(
                    "conv{}".format(i),
                    conv(settings[i - 1], depth, kernel_size=3, stride=2, **extra_args),
                )

    def forward(self, xx):
        out = []
        x = xx[0]
        for i in range(len(xx)):
            if i != 0:
                x = getattr(self, "conv{}".format(i))(x) + xx[i]
            x = getattr(self, "block{}".format(i))(x)
            out.append(x)
        return out[::-1]


class Head(nn.Sequential):
    def __init__(self, in_channels, out_planes):
        super(Head, self).__init__(
            ConvBNReLU(in_channels, in_channels, 3),
            nn.Conv2d(in_channels, out_planes, 3, padding=1),
        )


class SSDShelf(SSDSBase):
    """ShelfNet for Fast Semantic Segmentation
    See: https://arxiv.org/pdf/1811.11254.pdf for more details.

    Args:
        backbone: backbone layers for input
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
        num_classes: num of classes 
    """

    def __init__(self, backbone, extras, head, num_classes):
        super(SSDShelf, self).__init__(backbone, num_classes)

        self.transforms = nn.ModuleList(extras[0])
        self.shelf_head = nn.Sequential(extras[1])
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        self.initialize()

    def initialize(self):
        self.backbone.initialize()
        self.transforms.apply(self.initialize_extra)
        self.shelf_head.apply(self.initialize_extra)
        self.loc.apply(self.initialize_head)
        self.conf.apply(self.initialize_head)
        for c in self.conf:
            c[-1].apply(self.initialize_prior)

    def forward(self, x):
        loc, conf = [list() for _ in range(2)]

        # apply bases layers and cache source layer outputs
        features = self.backbone(x)

        features_len = len(features)
        features = [self.transforms[i](x) for i, x in enumerate(features)]

        features = self.shelf_head(features[::-1])
        for i in range(len(features), len(self.transforms)):
            features.append(self.transforms[i](features[-1]))

        # apply multibox head to source layers
        for (x, l, c) in zip(features, self.loc, self.conf):
            loc.append(l(x))
            conf.append(c(x))

        if not self.training:
            conf = [c.sigmoid() for c in conf]
        return tuple(loc), tuple(conf)

    @staticmethod
    def add_extras(feature_layer, mbox, num_classes):
        nets_outputs, transform_layers, loc_layers, conf_layers = [
            list() for _ in range(4)
        ]
        shelf_depths = []
        for layer, depth, box in zip(feature_layer[0], feature_layer[1], mbox):
            if isinstance(layer, int):
                if isinstance(depth, list):
                    if len(depth) == 2:
                        in_channels = depth[0]
                        depth = depth[1]
                else:
                    in_channels = depth
                nets_outputs.append(layer)
                shelf_depths.append(in_channels)
                transform_layers += [nn.Conv2d(in_channels, depth, 1)]
            else:
                transform_layers += parse_feature_layer(layer, in_channels, depth)
                in_channels = depth

            loc_layers += [Head(in_channels, box * 4)]
            conf_layers += [Head(in_channels, box * num_classes)]

        shelf_head = OrderedDict(
            [
                ("decoder0", ShelfPyramid(shelf_depths[::-1])),
                ("encoder0", ShelfPyramid(shelf_depths, conv=ConvBNReLU)),
                ("decoder1", ShelfPyramid(shelf_depths[::-1])),
            ]
        )
        return nets_outputs, (transform_layers, shelf_head), (loc_layers, conf_layers)
