import torch
import torch.nn as nn
import torch.nn.functional as F

import os

from ssds.modeling.layers.basic_layers import _conv
from ssds.modeling.layers.layers_parser import parse_feature_layer


class FSSD(nn.Module):
    """FSSD: Feature Fusion Single Shot Multibox Detector
    See: https://arxiv.org/pdf/1712.00960.pdf for more details.

    Args:
        phase: (string) Can be "eval" or "train" or "feature"
        base: base layers for input
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
        features： include to feature layers to fusion feature and build pyramids
        feature_layer: the feature layers for head to loc and conf
        num_classes: num of classes 
    """

    def __init__(self, base, extras, head, features, feature_layer, num_classes):
        super(FSSD, self).__init__()
        self.num_classes = num_classes

        # FSSD network
        self.base = nn.ModuleList(base)
        self.extras = nn.ModuleList(extras)
        self.feature_layer = feature_layer[0][0]
        self.transforms = nn.ModuleList(features[0])
        self.pyramids = nn.ModuleList(features[1])
        self.norm = nn.BatchNorm2d(
            int(feature_layer[0][1][-1] / 2) * len(self.transforms), affine=True
        )

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, phase="eval"):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]

            feature:
                the features maps of the feature extractor
        """
        sources, transformed, pyramids, loc, conf = [list() for _ in range(5)]

        # apply bases layers and cache source layer outputs
        for k in range(len(self.base)):
            x = self.base[k](x)
            if k in self.feature_layer:
                sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = v(x)
            sources.append(x)

        assert len(self.transforms) == len(sources)
        upsize = (sources[0].size()[2], sources[0].size()[3])

        for k, v in enumerate(self.transforms):
            size = None if k == 0 else upsize
            transformed.append(v(sources[k], size))
        x = torch.cat(transformed, 1)
        x = self.norm(x)
        for k, v in enumerate(self.pyramids):
            x = v(x)
            pyramids.append(x)

        if phase == "feature":
            return pyramids

        # apply multibox head to pyramids layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).view(x.size(0), 4, -1))
            conf.append(c(x).view(x.size(0), self.num_classes, -1))
        loc = torch.cat(loc, 2).contiguous()
        conf = torch.cat(conf, 2).contiguous()

        return loc, conf


class BasicConvWithUpSample(nn.Module):
    # temp, need TODO improve
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        relu=True,
        bn=False,
        bias=True,
    ):
        super(BasicConvWithUpSample, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = (
            nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            if bn
            else None
        )
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x, up_size=None):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        if up_size is not None:
            x = F.upsample(x, size=up_size, mode="bilinear")
        return x


def add_extras(base, feature_layer, mbox, num_classes, version):
    extra_layers = []
    feature_transform_layers = []
    pyramid_feature_layers = []
    loc_layers = []
    conf_layers = []
    in_channels = None

    feature_transform_channel = int(feature_layer[0][1][-1] / 2)
    for layer, depth in zip(feature_layer[0][0], feature_layer[0][1]):
        extra_layers += parse_feature_layer(layer, in_channels, depth)
        in_channels = depth
        feature_transform_layers += [
            BasicConvWithUpSample(
                in_channels, feature_transform_channel, kernel_size=1, padding=0
            )
        ]

    in_channels = len(feature_transform_layers) * feature_transform_channel
    for layer, depth, box in zip(feature_layer[1][0], feature_layer[1][1], mbox):
        pyramid_feature_layers += parse_feature_layer(layer, in_channels, depth)
        in_channels = depth
        loc_layers += [nn.Conv2d(in_channels, box * 4, kernel_size=3, padding=1)]
        conf_layers += [
            nn.Conv2d(in_channels, box * num_classes, kernel_size=3, padding=1)
        ]
    return (
        base,
        extra_layers,
        (feature_transform_layers, pyramid_feature_layers),
        (loc_layers, conf_layers),
    )


def build_fssd(base, feature_layer, mbox, num_classes):
    base_, extras_, features_, head_ = add_extras(
        base(), feature_layer, mbox, num_classes, version="fssd"
    )
    return FSSD(base_, extras_, head_, features_, feature_layer, num_classes)
