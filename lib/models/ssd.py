import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import os

from lib.layers import *

class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, base, extras, head, feature_layer, num_classes):
        super(SSD, self).__init__()
        self.num_classes = num_classes
        # SSD network
        self.base = nn.ModuleList(base)
        self.extras = nn.ModuleList(extras)
        self.feature_layer = feature_layer
        # print(self.base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.norm = L2Norm(512, 20)
        # print(self.extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        # print(self.loc)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, is_train = False):
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
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()

        # apply bases layers and cache source layer outputs
        for k in range(len(self.base)):
            x = self.base[k](x)
            if k in self.feature_layer:
                if len(sources) == 0:
                    s = self.norm(x)
                    sources.append(s)
                else:
                    sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        # print([o.size() for o in loc])
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if is_train == False:
            output = (
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(-1, self.num_classes)),  # conf preds
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                #self.priors
            )
        return output

    def load_weights(self, resume_checkpoint, resume_scope=''):
        if os.path.isfile(resume_checkpoint):
            print(("=> loading checkpoint '{}'".format(resume_checkpoint)))
            checkpoint = torch.load(resume_checkpoint)

            # remove the module in the parrallel model
            if 'module.' in list(checkpoint.items())[0][0]: 
                pretrained_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint.items())}
                checkpoint = pretrained_dict   

            # change L2Norm to norm
            for k, v in list(checkpoint.items()):
                if 'L2Norm' in k:
                    k.replace('L2Norm','norm')

            # extract the weights based on the resume scope
            if resume_scope !='':
                pretrained_dict = {}
                if resume_scope == 'classfication':
                    # TODO: load weight from pretrain classification
                    print('TODO: load weight from pretrain classification')
                else:
                    for k, v in list(checkpoint.items()):
                        for resume_key in resume_scope.split(','):
                            if resume_key in k:
                                pretrained_dict[k] = v
                                break
                checkpoint = pretrained_dict         

            pretrained_dict = {k: v for k, v in checkpoint.items() if k in self.state_dict()}
            checkpoint = self.state_dict()
            checkpoint.update(pretrained_dict) 
            # print([k for k, v in list(checkpoint.items())])

            self.load_state_dict(checkpoint)

        else:
            print(("=> no checkpoint found at '{}'".format(resume_checkpoint)))

    def _forward_features_size(self, img_size):
        x = torch.rand(1, 3, img_size[0], img_size[1])
        x = torch.autograd.Variable(x, volatile=True).cuda()
        sources = list()
        self.eval()
        # apply bases layers and cache source layer outputs
        for k in range(len(self.base)):
            x = self.base[k](x)
            if k in self.feature_layer:
                if len(sources) == 0:
                    s = self.norm(x)
                    sources.append(s)
                else:
                    sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        return [(o.size()[2], o.size()[3]) for o in sources]

def add_extras(base, feature_layer, layer_depth, mbox, num_classes):
    extra_layers = []
    loc_layers = []
    conf_layers = []
    in_channels = None
    for layer, depth, box in zip(feature_layer, layer_depth, mbox):
        if layer == 'S':
            extra_layers += [
                    nn.Conv2d(in_channels, int(depth/2), kernel_size=1),
                    nn.Conv2d(int(depth/2), depth, kernel_size=3, stride=2, padding=1)  ]
            in_channels = depth
        elif layer == '':
            extra_layers += [
                    nn.Conv2d(in_channels, int(depth/2), kernel_size=1),
                    nn.Conv2d(int(depth/2), depth, kernel_size=3)  ]
            in_channels = depth
        else:
            in_channels = depth
        loc_layers += [nn.Conv2d(in_channels, box * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(in_channels, box * num_classes, kernel_size=3, padding=1)]
    return base, extra_layers, (loc_layers, conf_layers)

def build_ssd(base, feature_layer, layer_depth, mbox, num_classes):
    base_, extras_, head_ = add_extras(base(), feature_layer, layer_depth, mbox, num_classes)
    return SSD(base_, extras_, head_, feature_layer, num_classes)