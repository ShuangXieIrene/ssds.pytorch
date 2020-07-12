import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from .ssdsbase import SSDSBase
from ssds.modeling.layers.basic_layers import ConvBNReLU, ConvBNReLUx2


class YOLOV3(SSDSBase):
    """YOLO V3 Architecture
    See: https://arxiv.org/pdf/1804.02767.pdf for more details.

    Args:
        base: base layers for input
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
        feature_layer: the feature layers for head to loc and conf
        num_classes: num of classes 
    """

    def __init__(self, backbone, extras, head, num_classes):
        super(YOLOV3, self).__init__(backbone, num_classes)

        self.transforms = nn.ModuleList(extras[0])
        self.extras = nn.ModuleList(extras[1])
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        self.initialize()

    def initialize(self):
        self.backbone.initialize()
        self.transforms.apply(self.initialize_extra)
        self.extras.apply(self.initialize_extra)
        self.loc.apply(self.initialize_head)
        self.conf.apply(self.initialize_head)
        for c in self.conf:
            c[-1].apply(self.initialize_prior)

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
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]

            feature:
                the features maps of the feature extractor
        """
        loc, conf = [list() for _ in range(2)]

        # apply backbone to input and cache outputs
        features = self.backbone(x)

        x = features[-1]
        xx = features[-1]
        features_len = len(features)
        for i in range(len(features))[::-1]:
            if i != features_len - 1:
                xx = F.interpolate(self.transforms[i](xx), scale_factor=2)
                xx = torch.cat((features[i], xx), dim=1)
            xx = self.extras[i](xx)
            features[i] = xx

        # apply multibox head to source layers
        for i, (l, c) in enumerate(zip(self.loc, self.conf)):
            if i < features_len:
                xx = features[i]
            elif i == features_len:
                xx = self.extras[i](x)
            else:
                xx = self.extras[i](xx)
            loc.append(l(xx))
            conf.append(c(xx))

        if not self.training:
            conf = [c.sigmoid() for c in conf]
        return tuple(loc), tuple(conf)

    @staticmethod
    def add_extras(feature_layer, mbox, num_classes):
        nets_outputs, transform_layers, extra_layers, loc_layers, conf_layers = [
            list() for _ in range(5)
        ]

        last_int_layer = [
            layer for layer in feature_layer[0] if isinstance(layer, int)
        ][-1]

        for layer, depth, box in zip(feature_layer[0], feature_layer[1], mbox):
            if isinstance(layer, int):
                nets_outputs.append(layer)
                if layer == last_int_layer:
                    if isinstance(depth, list):
                        extra_layers += [ConvBNReLUx2(depth[0], depth[1], 3)]
                    else:
                        extra_layers += [ConvBNReLUx2(depth, depth // 2, 3)]
                else:
                    prev_depth = feature_layer[1][feature_layer[0].index(layer) + 1]
                    if isinstance(depth, list):
                        transform_layers += [
                            ConvBNReLU(prev_depth[1], depth[0] // 2, 3)
                        ]
                        extra_layers += [ConvBNReLUx2(int(depth[0] * 1.5), depth[1], 3)]
                    else:
                        transform_layers += [ConvBNReLU(prev_depth // 2, depth // 2, 3)]
                        extra_layers += [ConvBNReLUx2(int(depth * 1.5), depth // 2, 3)]
            elif layer == "Conv:S":
                extra_layers += [ConvBNReLU(in_channels, depth, 3, stride=2)]
            else:
                raise ValueError(layer + " does not support by YOLO")
            in_channels = (
                depth[1]
                if isinstance(depth, list)
                else depth // 2
                if isinstance(layer, int)
                else depth
            )
            loc_layers += [
                nn.Sequential(
                    ConvBNReLU(in_channels, in_channels, 3),
                    nn.Conv2d(in_channels, box * 4, kernel_size=3, padding=1),
                )
            ]
            conf_layers += [
                nn.Sequential(
                    ConvBNReLU(in_channels, in_channels, 3),
                    nn.Conv2d(in_channels, box * num_classes, kernel_size=3, padding=1),
                )
            ]
            in_channels = depth[0] if isinstance(depth, list) else depth
        return nets_outputs, (transform_layers, extra_layers), (loc_layers, conf_layers)


class SPPModule(nn.Module):
    def __init__(self, num_levels, pool_type="max_pool"):
        super(SPPModule, self).__init__()

        self.num_levels = num_levels
        self.pool_type = pool_type

    def forward(self, x):
        bs, c, h, w = x.size()
        pooling_layers = [x]
        for i in range(self.num_levels):
            kernel_size = 4 * (i + 1) + 1
            padding = (kernel_size - 1) // 2
            if self.pool_type == "max_pool":
                tensor = F.max_pool2d(
                    x, kernel_size=kernel_size, stride=1, padding=padding
                )
            else:
                tensor = F.avg_pool2d(
                    x, kernel_size=kernel_size, stride=1, padding=padding
                )
            pooling_layers.append(tensor)
        x = torch.cat(pooling_layers, dim=1)
        return x


class PANModule(nn.Module):
    def __init__(self, channels):
        super(PANModule, self).__init__()

        self.levels = len(channels)

        for i in range(self.levels - 1, 0, -1):
            self.add_module(
                "top-down-{}-to-{}".format(i, i - 1),
                ConvBNReLU(channels[i], channels[i - 1]),
            )
            self.add_module(
                "top-down-{}".format(i - 1),
                ConvBNReLUx2(channels[i - 1] * 2, channels[i - 1]),
            )

        for i in range(0, self.levels - 1, 1):
            self.add_module(
                "bottom-up-{}-to-{}".format(i, i + 1),
                ConvBNReLU(channels[i], channels[i + 1], stride=2),
            )
            self.add_module(
                "bottom-up-{}".format(i + 1),
                ConvBNReLUx2(channels[i + 1] * 2, channels[i + 1]),
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, val=0)

    def forward(self, xx):
        assert len(xx) == self.levels

        # build top-down
        for i in range(self.levels - 1, 0, -1):
            xx[i - 1] = torch.cat(
                (
                    xx[i - 1],
                    F.interpolate(
                        getattr(self, "top-down-{}-to-{}".format(i, i - 1))(xx[i]),
                        scale_factor=2,
                        mode="nearest",
                    ),
                ),
                dim=1,
            )
            xx[i - 1] = getattr(self, "top-down-{}".format(i - 1))(xx[i - 1])

        # build bottom-up
        for i in range(0, self.levels - 1, 1):
            xx[i + 1] = torch.cat(
                (
                    xx[i + 1],
                    getattr(self, "bottom-up-{}-to-{}".format(i, i + 1))(xx[i]),
                ),
                dim=1,
            )
            xx[i + 1] = getattr(self, "bottom-up-{}".format(i + 1))(xx[i + 1])
        return xx


class YOLOV4(SSDSBase):
    """YOLO V3 Architecture
    See: https://arxiv.org/pdf/1804.02767.pdf for more details.

    Args:
        base: base layers for input
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
        feature_layer: the feature layers for head to loc and conf
        num_classes: num of classes 
    """

    def __init__(self, backbone, extras, head, num_classes):
        super(YOLOV4, self).__init__(backbone, num_classes)

        self.transforms = nn.ModuleList(extras[0])
        self.extras = nn.ModuleList(extras[1])
        self.fpn = extras[2]
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        self.initialize()

    def initialize(self):
        self.backbone.initialize()
        self.transforms.apply(self.initialize_extra)
        self.fpn.apply(self.initialize_extra)
        self.extras.apply(self.initialize_extra)
        self.loc.apply(self.initialize_head)
        self.conf.apply(self.initialize_head)
        for c in self.conf:
            c[-1].apply(self.initialize_prior)

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
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]

            feature:
                the features maps of the feature extractor
        """
        loc, conf = [list() for _ in range(2)]

        # apply backbone to input and cache outputs
        features = self.backbone(x)

        # tansmit + fpn
        for i, t in enumerate(self.transforms):
            features[i] = t(features[i])
        features = self.fpn(features)

        # extra layers
        x = features[-1]
        for e in self.extras:
            x = e(x)
            features.append(x)

        # apply multibox head to source layers
        for i, (l, c) in enumerate(zip(self.loc, self.conf)):
            loc.append(l(features[i]))
            conf.append(c(features[i]))

        if not self.training:
            conf = [c.sigmoid() for c in conf]
        return tuple(loc), tuple(conf)

    @staticmethod
    def add_extras(feature_layer, mbox, num_classes):
        nets_outputs, transform_layers, extra_layers, loc_layers, conf_layers = [
            list() for _ in range(5)
        ]

        last_int_layer = [
            layer for layer in feature_layer[0] if isinstance(layer, int)
        ][-1]
        fpn_channels = []
        for layer, depth, box in zip(feature_layer[0], feature_layer[1], mbox):
            if isinstance(layer, int):
                nets_outputs.append(layer)
                fpn_channels.append(depth // 2)
                if layer == last_int_layer:
                    transform_layers += [
                        nn.Sequential(
                            ConvBNReLU(depth, depth // 2, 3),
                            SPPModule(3),
                            ConvBNReLU(depth * 2, depth // 2, 3),
                        )
                    ]
                else:
                    transform_layers += [
                        ConvBNReLU(depth, depth // 2, 3)
                    ]  # [ConvBNReLU(depth, 256, 1)]
            elif layer == "Conv:S":
                extra_layers += [
                    ConvBNReLU(in_channels, depth, 3, stride=2)
                ]  # [nn.Conv2d(depth, 256, 3, stride=2, padding=1)]
            else:
                raise ValueError(layer + " does not support by YOLO")
            in_channels = depth // 2 if isinstance(layer, int) else depth
            loc_layers += [
                nn.Sequential(
                    ConvBNReLU(in_channels, in_channels, 3),
                    nn.Conv2d(in_channels, box * 4, kernel_size=3, padding=1),
                )
            ]
            conf_layers += [
                nn.Sequential(
                    ConvBNReLU(in_channels, in_channels, 3),
                    nn.Conv2d(in_channels, box * num_classes, kernel_size=3, padding=1),
                )
            ]

        num_stack = 1 if len(feature_layer) == 2 else feature_layer[2]
        fpn = nn.Sequential(*[PANModule(fpn_channels) for _ in range(num_stack)])
        return (
            nets_outputs,
            (transform_layers, extra_layers, fpn),
            (loc_layers, conf_layers),
        )


if __name__ == "__main__":
    model = PANModule([2, 4, 8], 3)
    model.eval()
    print(model)
    xx = [torch.ones(1, 2 ** i, 2 ** (4 - i), 2 ** (4 - i)) * i for i in range(1, 4)]
    model(xx)

    torch.onnx.export(model, xx, "test.onnx")
