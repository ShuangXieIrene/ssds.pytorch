import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import os

from lib.layers import *

class Retina(nn.Module):
    def __init__(self, base, extras, head, feature_layer, num_classes):
        super(Retina, self).__init__()
        self.num_classes = num_classes
        # SSD network
        self.base = nn.ModuleList(base)
        self.extras = nn.ModuleList(extras[1])
        self.transforms = nn.ModuleList(extras[0])
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        self.softmax = nn.Softmax(dim=-1)

        self.feature_layer = feature_layer[0]
    
    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y    

    def forward(self, x, phase='eval'):
        sources, loc, conf = [list() for _ in range(3)]

        # apply bases layers and cache source layer outputs
        for k in range(len(self.base)):
            x = self.base[k](x)
            if k in self.feature_layer:
                sources.append(x)

        for i in range(len(sources))[::-1]:
            if i != len(sources) -1:
                xx = self.extras[i](self._upsample_add(xx, self.transforms[i](sources[i])))
            else:
                xx = self.transforms[i](sources[i])
            sources[i] = xx

        # apply extra layers and cache source layer outputs
        for i, v in enumerate(self.extras):
            if i >= len(sources):
                x = v(x)
                sources.append(x)

        if phase == 'feature':
            return sources

        # apply multibox head to source layers
        for x in sources:
            loc.append(self.loc(x).permute(0, 2, 3, 1).contiguous())
            conf.append(self.conf(x).permute(0, 2, 3, 1).contiguous())
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if phase == 'eval':
            output = (
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(-1, self.num_classes)),  # conf preds
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
            )
        return output
        


def add_extras(base, feature_layer, mbox, num_classes, version):
    extra_layers = []
    transform_layers = []
    loc_layers = [Retina_head(box * 4)]
    conf_layers = [Retina_head(box * num_classes)]

    for layer, in_channels, box in zip(feature_layer[0], feature_layer[1], mbox):
        if 'lite' in version:
            if layer == 'S':
                extra_layers += [ _conv_dw(in_channels, 256, stride=2, padding=1, expand_ratio=1) ]
            elif layer == '':
                extra_layers += [ _conv_dw(in_channels, 256, stride=1, expand_ratio=1) ]
            else:
                extra_layers += [ _conv_dw(256, 256, stride=1, padding=1, expand_ratio=1) ]
                transform_layers += [ _conv_pw(in_channels, 256) ]
        else:    
            if layer == 'S':
                extra_layers += [ _conv(in_channels, 256, stride=2, padding=1) ]
            elif layer == '':
                extra_layers += [ _conv(in_channels, 256, stride=1) ]
            else:
                extra_layers += [ _conv(256, 256, stride=1, padding=1) ]
                transform_layers += [ _conv_pw(in_channels, 256) ]
    return base, (transform_layers, extra_layers), (loc_layers, conf_layers)

def Retina_head(self, out_planes):
    layers = []
    for _ in range(4):
        layers.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU(True))
    layers.append(nn.Conv2d(256, out_planes, kernel_size=3, stride=1, padding=1))
    return nn.Sequential(*layers)

# based on the implementation in https://github.com/tensorflow/models/blob/master/research/object_detection/models/feature_map_generators.py#L213
# when the expand_ratio is 1, the implemetation is nearly same. Since the shape is always change, I do not add the shortcut as what mobilenetv2 did.
def _conv_dw(inp, oup, stride=1, padding=0, expand_ratio=1):
    return nn.Sequential(
        # pw
        nn.Conv2d(inp, oup * expand_ratio, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup * expand_ratio),
        nn.ReLU6(inplace=True),
        # dw
        nn.Conv2d(oup * expand_ratio, oup * expand_ratio, 3, stride, padding, groups=oup * expand_ratio, bias=False),
        nn.BatchNorm2d(oup * expand_ratio),
        nn.ReLU6(inplace=True),
        # pw-linear
        nn.Conv2d(oup * expand_ratio, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
    )

def _conv_pw(inp, oup, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
    )


def _conv(inp, oup, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, padding, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )


def build_retina(base, feature_layer, mbox, num_classes):
    """RetinaNet in Focal Loss for Dense Object Detection
    See: https://arxiv.org/pdf/1708.02002.pdffor more details.
    """
    base_, extras_, head_ = add_extras(base(), feature_layer, mbox, num_classes, version='retinanet')
    return Retina(base_, extras_, head_, feature_layer, num_classes)

def build_retina_lite(base, feature_layer, mbox, num_classes):
    """RetinaNet in Focal Loss for Dense Object Detection
    See: https://arxiv.org/pdf/1708.02002.pdffor more details.
    """
    base_, extras_, head_ = add_extras(base(), feature_layer, mbox, num_classes, version='retinanet_lite')
    return SSD(base_, extras_, head_, feature_layer, num_classes)