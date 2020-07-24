import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from .ssdsbase import SSDSBase
from ssds.modeling.layers.layers_parser import parse_feature_layer


class SSD(SSDSBase):
    r"""SSD: Single Shot MultiBox Detector
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        backbone: backbone layers for input
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
        num_classes: num of classes 
    """

    def __init__(self, backbone, extras, head, num_classes):
        super(SSD, self).__init__(backbone, num_classes)

        # SSD head
        self.extras = nn.ModuleList(extras)
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        self.initialize()

    def initialize(self):
        r"""
        :meta private:
        """
        self.backbone.initialize()
        self.extras.apply(self.initialize_extra)
        self.loc.apply(self.initialize_head)
        self.conf.apply(self.initialize_head)
        for c in self.conf:
            c.apply(self.initialize_prior)

    def forward(self, x):
        r"""Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images.

        Return:
            When self.training==True, loc and conf for each anchor box;

            When self.training==False. loc and conf.sigmoid() for each anchor box;

            For each player, conf with shape [batch, num_anchor*num_classes, height, width];

            For each player, loc  with shape [batch, num_anchor*4, height, width].
        """
        loc, conf = [list() for _ in range(2)]

        # apply backbone to input and cache outputs
        features = self.backbone(x)

        # apply extra blocks and cache outputs
        for v in self.extras:
            x = v(features[-1])
            features.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(features, self.loc, self.conf):
            loc.append(l(x))
            conf.append(c(x))

        if not self.training:
            conf = [c.sigmoid() for c in conf]
        return tuple(loc), tuple(conf)

    @staticmethod
    def add_extras(feature_layer, mbox, num_classes):
        r"""Define and declare the extras, loc and conf modules for the ssd model.

        The feature_layer is defined in cfg.MODEL.FEATURE_LAYER. For ssd model can be int, list of int and str:

        * int
            The int in the feature_layer represents the output feature in the backbone.
        * str
            The str in the feature_layer represents the extra layers append at the end of the backbone.

        Args:
            feature_layer: the feature layers with detection head, defined by cfg.MODEL.FEATURE_LAYER
            mbox: the number of boxes for each feature map
            num_classes: the number of classes, defined by cfg.MODEL.NUM_CLASSES
        """
        nets_outputs, extra_layers, loc_layers, conf_layers = [list() for _ in range(4)]
        in_channels = None
        for layer, depth, box in zip(feature_layer[0], feature_layer[1], mbox):
            if isinstance(layer, int):
                nets_outputs.append(layer)
            else:
                extra_layers += parse_feature_layer(layer, in_channels, depth)
            in_channels = depth
            loc_layers += [nn.Conv2d(in_channels, box * 4, kernel_size=3, padding=1)]
            conf_layers += [
                nn.Conv2d(in_channels, box * num_classes, kernel_size=3, padding=1)
            ]
        return nets_outputs, extra_layers, (loc_layers, conf_layers)
