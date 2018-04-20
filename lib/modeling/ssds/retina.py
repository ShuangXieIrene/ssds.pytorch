import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import os

from lib.layers import *

class Retina(nn.Module):
    def __init__(self, base, extras, norm, head, feature_layer, num_classes):
        super(RFB, self).__init__()
        pass

    def forward(self, x, phase='eval'):
        pass


def add_extras(base, feature_layer, mbox, num_classes, version):
    pass

def build_retina(base, feature_layer, mbox, num_classes):
    pass

def buid_retina_lite(base, feature_layer, mbox, num_classes):
    pass