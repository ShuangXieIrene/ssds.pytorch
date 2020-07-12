""" This file is similar with the effcientnet file, but use torch hub instand of using 
"""

import torch
import torch.nn as nn
from .rutils import register


class EffNet(nn.Module):
    def __init__(self, model_name, outputs, exportable=False, **kwargs):
        super(EffNet, self).__init__()
        self.outputs = outputs

        if exportable:
            import geffnet

            geffnet.config.set_exportable(True)
            model = geffnet.create_model(model_name, **kwargs)
        else:
            model = torch.hub.load(
                "rwightman/gen-efficientnet-pytorch", model_name, **kwargs
            )

        self.conv_stem = model.conv_stem
        self.bn1 = model.bn1
        self.act1 = model.act1
        for j in range(7):
            self.add_module(
                "block{}".format(j + 1), getattr(model.blocks, "{}".format(j))
            )

    def forward(self, x):
        x = self.act1(self.bn1(self.conv_stem(x)))

        outputs = []
        for level in range(1, 8):
            # level = j + 1 # only 1 conv before
            if level > max(self.outputs):
                break
            x = getattr(self, "block{}".format(level))(x)
            if level in self.outputs:
                outputs.append(x)

        return outputs

    def initialize(self):
        pass


@register
def EffNetB0(outputs, **kwargs):
    return EffNet("efficientnet_b0", outputs, drop_connect_rate=0.2, pretrained=True)


@register
def EffNetB1(outputs, **kwargs):
    return EffNet("efficientnet_b1", outputs, drop_connect_rate=0.2, pretrained=True)


@register
def EffNetB2(outputs, **kwargs):
    return EffNet("efficientnet_b2", outputs, drop_connect_rate=0.2, pretrained=True)


@register
def EffNetB3(outputs, **kwargs):
    return EffNet("efficientnet_b3", outputs, drop_connect_rate=0.2, pretrained=True)


@register
def EffNetB4(outputs, **kwargs):
    return EffNet("efficientnet_b4", outputs, drop_connect_rate=0.2, pretrained=True)


@register
def EffNetB5(outputs, **kwargs):
    return EffNet("efficientnet_b5", outputs, drop_connect_rate=0.2, pretrained=True)


@register
def EffNetB6(outputs, **kwargs):
    return EffNet("efficientnet_b6", outputs, drop_connect_rate=0.2, pretrained=True)


@register
def EffNetB7(outputs, **kwargs):
    return EffNet("efficientnet_b7", outputs, drop_connect_rate=0.2, pretrained=True)


@register
def EffNetB0Ex(outputs, **kwargs):
    return EffNet(
        "efficientnet_b0",
        outputs,
        drop_connect_rate=0.2,
        pretrained=True,
        exportable=True,
    )
