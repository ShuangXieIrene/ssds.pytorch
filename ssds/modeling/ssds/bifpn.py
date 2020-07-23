import torch
import torch.nn as nn
import torch.nn.functional as F

from .ssdsbase import SSDSBase
from .fpn import SharedHead
from ssds.modeling.layers.basic_layers import ConvBNReLU, SepConvBNReLU


class BiFPNModule(nn.Module):
    def __init__(self, channels, levels, init=0.5, block=ConvBNReLU):
        super(BiFPNModule, self).__init__()

        self.levels = levels
        self.w1 = nn.Parameter(torch.Tensor(2, levels).fill_(init))
        self.w2 = nn.Parameter(torch.Tensor(3, levels - 2).fill_(init))

        for i in range(levels - 1, 0, -1):
            self.add_module("top-down-{}".format(i - 1), block(channels, channels))

        for i in range(0, levels - 1, 1):
            self.add_module("bottom-up-{}".format(i + 1), block(channels, channels))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, val=0)

    def forward(self, xx):
        assert len(xx) == self.levels
        levels = self.levels

        # normalize weights
        w1 = F.relu(self.w1)
        w1 /= torch.sum(w1, dim=0) + 1e-6
        w2 = F.relu(self.w2)
        w2 /= torch.sum(w2, dim=0) + 1e-6

        # build top-down
        xs = [[]] + [x for x in xx[1:-1]] + [[]]
        for i in range(levels - 1, 0, -1):
            xx[i - 1] = w1[0, i - 1] * xx[i - 1] + w1[1, i - 1] * F.interpolate(
                xx[i], scale_factor=2, mode="nearest"
            )
            xx[i - 1] = getattr(self, "top-down-{}".format(i - 1))(xx[i - 1])

        # build bottom-up
        for i in range(0, levels - 2, 1):
            xx[i + 1] = (
                w2[0, i] * xx[i + 1]
                + w2[1, i] * F.max_pool2d(xx[i], kernel_size=2)
                + w2[2, i] * xs[i + 1]
            )
            xx[i + 1] = getattr(self, "bottom-up-{}".format(i + 1))(xx[i + 1])

        xx[levels - 1] = w1[0, levels - 1] * xx[levels - 1] + w1[
            1, levels - 1
        ] * F.max_pool2d(xx[levels - 2], kernel_size=2)
        xx[levels - 1] = getattr(self, "bottom-up-{}".format(levels - 1))(
            xx[levels - 1]
        )
        return xx


class SSDBiFPN(SSDSBase):
    """EfficientDet: Scalable and Efficient Object Detection
    See: https://arxiv.org/abs/1911.09070v6 for more details.

    Compared with the original implementation, change the conv2d 
    in the extra and head to ConvBNReLU to helps the model converage easily
    Not add the bn&relu to transforms cause it is followed by interpolate and element-wise sum

    Args:
        backbone: backbone layers for input
        extras: contains transforms, extra and stack_bifpn layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
        num_classes: num of classes 
    """

    def __init__(self, backbone, extras, head, num_classes):
        super(SSDBiFPN, self).__init__(backbone, num_classes)

        # SSD network
        self.transforms = nn.ModuleList(extras[0])
        self.extras = nn.ModuleList(extras[1])
        self.stack_bifpn = extras[2]
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
        for i in range(features_len):
            features[i] = self.transforms[i](features[i])
        features = self.stack_bifpn(features)

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
        transform_layers = []
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
        num_stack = 1 if len(feature_layer) == 2 else feature_layer[2]
        fpn = nn.Sequential(
            *[BiFPNModule(256, len(transform_layers)) for _ in range(num_stack)]
        )
        return (
            nets_outputs,
            (transform_layers, extra_layers, fpn),
            (loc_layers, conf_layers),
        )


if __name__ == "__main__":
    model = BiFPNModule(1, 4)
    model.eval()
    xx = [torch.ones(1, 1, i, i) * i for i in [8, 4, 2, 1]]
    model(xx)

    torch.onnx.export(model, xx, "test.onnx")
