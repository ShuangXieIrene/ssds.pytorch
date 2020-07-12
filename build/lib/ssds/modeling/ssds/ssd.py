import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from .ssdsbase import SSDSBase
from ssds.modeling.layers.layers_parser import parse_feature_layer


class SSD(SSDSBase):
    """Single Shot Multibox Architecture
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
        self.backbone.initialize()
        self.extras.apply(self.initialize_extra)
        self.loc.apply(self.initialize_head)
        self.conf.apply(self.initialize_head)
        for c in self.conf:
            c.apply(self.initialize_prior)

    def forward(self, x):
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
                    1: confidence layers, Shape: [batch, num_priors*num_classes, height, width]
                    2: localization layers, Shape: [batch, num_priors*4, height, width]

            feature:
                the features maps of the feature extractor
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
