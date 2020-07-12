import torch
import torch.nn as nn
import math


class SSDSBase(nn.Module):
    def __init__(self, backbone, num_classes):
        super(SSDSBase, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes

    # Initialize class head prior
    def initialize_prior(self, layer):
        pi = 0.01
        b = -math.log((1 - pi) / pi)
        nn.init.constant_(layer.bias, b)
        nn.init.normal_(layer.weight, std=0.01)

    def initialize_head(self, layer):
        if isinstance(layer, nn.Conv2d):
            nn.init.normal_(layer.weight, std=0.01)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, val=0)

    def initialize_extra(self, layer):
        if isinstance(layer, nn.Conv2d):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, val=0)
