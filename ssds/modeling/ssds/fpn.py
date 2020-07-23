import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ssdsbase import SSDSBase
from ssds.modeling.layers.basic_layers import ConvBNReLU


class SharedHead(nn.Sequential):
    def __init__(self, out_planes):
        layers = []
        for _ in range(4):
            layers += [
                ConvBNReLU(256, 256, 3)
            ]  # [nn.Conv2d(256, 256, 3, padding=1), nn.ReLU()]
        layers += [nn.Conv2d(256, out_planes, 3, padding=1)]
        super(SharedHead, self).__init__(*layers)


class SSDFPN(SSDSBase):
    """RetinaNet in Focal Loss for Dense Object Detection
    See: https://arxiv.org/abs/1708.02002v2 for more details.

    Compared with the original implementation, change the conv2d 
    in the extra and head to ConvBNReLU to helps the model converage easily
    Not add the bn&relu to transforms cause it is followed by interpolate and element-wise sum

    Args:
        backbone: backbone layers for input
        extras: contains transforms and extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
        num_classes: num of classes 
    """

    def __init__(self, backbone, extras, head, num_classes):
        super(SSDFPN, self).__init__(backbone, num_classes)

        # SSD network
        self.transforms = nn.ModuleList(extras[0])
        self.extras = nn.ModuleList(extras[1])
        self.loc = head[0]
        self.conf = head[1]

        self.initialize()

    def initialize(self):
        r"""
        :meta private:
        """
        self.backbone.initialize()
        self.transforms.apply(self.initialize_extra)
        self.extras.apply(self.initialize_extra)
        self.loc.apply(self.initialize_head)
        self.conf.apply(self.initialize_head)
        self.conf[-1].apply(self.initialize_prior)

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

        # apply bases layers and cache source layer outputs
        features = self.backbone(x)

        x = features[-1]
        features_len = len(features)
        for i in range(len(features))[::-1]:
            if i != features_len - 1:
                xx = F.interpolate(
                    xx, scale_factor=2, mode="nearest"
                ) + self.transforms[i](features[i])
            else:
                xx = self.transforms[i](features[i])
            features[i] = xx

        for i, v in enumerate(self.extras):
            if i < features_len:
                xx = v(features[i])
            elif i == features_len:
                xx = v(x)
            else:
                xx = v(xx)
            loc.append(self.loc(xx))
            conf.append(self.conf(xx))

        if not self.training:
            conf = [c.sigmoid() for c in conf]
        return tuple(loc), tuple(conf)

    @staticmethod
    def add_extras(feature_layer, mbox, num_classes):
        r"""Define and declare the extras, loc and conf modules for the ssdfpn model.

        The feature_layer is defined in cfg.MODEL.FEATURE_LAYER. For ssdfpn model can be int, list of int and str:

        * int
            The int in the feature_layer represents the output feature in the backbone.
        * list of int
            The list of int in the feature_layer represents the output feature in the backbone, the first int is the \
            backbone output and the second int is the upsampling branch to fuse feature.
        * str
            The str in the feature_layer represents the extra layers append at the end of the backbone.

        Args:
            feature_layer: the feature layers with detection head, defined by cfg.MODEL.FEATURE_LAYER
            mbox: the number of boxes for each feature map
            num_classes: the number of classes, defined by cfg.MODEL.NUM_CLASSES
        """

        nets_outputs, transform_layers, extra_layers = [list() for _ in range(3)]
        if not all(mbox[i] == mbox[i + 1] for i in range(len(mbox) - 1)):
            raise ValueError(
                "For SSDFPN module, the number of box have to be same in every layer"
            )
        loc_layers = SharedHead(mbox[0] * 4)
        conf_layers = SharedHead(mbox[0] * num_classes)

        for layer, depth in zip(feature_layer[0], feature_layer[1]):
            if isinstance(layer, int):
                nets_outputs.append(layer)
                transform_layers += [
                    nn.Conv2d(depth, 256, 1)
                ]  # [ConvBNReLU(depth, 256, 1)]
                extra_layers += [
                    ConvBNReLU(256, 256, 3)
                ]  # [nn.Conv2d(256, 256, 3, padding=1)]
            elif layer == "Conv:S":
                extra_layers += [
                    ConvBNReLU(depth, 256, 3, stride=2)
                ]  # [nn.Conv2d(depth, 256, 3, stride=2, padding=1)]
            else:
                raise ValueError(layer + " does not support by SSDFPN")
        return nets_outputs, (transform_layers, extra_layers), (loc_layers, conf_layers)
