import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import os

from lib.layers import *

class FSSD(nn.Module):
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

    def __init__(self, base, extras, head, features, feature_layer, num_classes):
        super(FSSD, self).__init__()
        self.num_classes = num_classes
        # SSD network
        self.base = nn.ModuleList(base)
        self.extras = nn.ModuleList(extras)
        self.feature_layer = feature_layer[0][0]
        self.transforms = nn.ModuleList(features[0])
        self.pyramids = nn.ModuleList(features[1])
        # print(self.base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.norm = nn.BatchNorm2d(256*len(self.transforms),affine=True)
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
        transformed = list()
        pyramids = list()
        loc = list()
        conf = list()

        # apply bases layers and cache source layer outputs
        for k in range(len(self.base)):
            x = self.base[k](x)
            if k in self.feature_layer:
                sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = v(x)
            if k % 2 == 1:
                sources.append(x)
        assert len(self.transforms) == len(sources)
        upsize = (sources[0].size()[2], sources[0].size()[3])
        for k, v in enumerate(self.transforms):
            size = None if k == 0 else upsize
            transformed.append(v(sources[k], size))
        x = torch.cat(transformed, 1)
        x = self.norm(x)
        for k, v in enumerate(self.pyramids):
            x = v(x)
            pyramids.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(pyramids, self.loc, self.conf):
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


            print("=> Weigths in the checkpoints:")
            print([k for k, v in list(checkpoint.items())])

            # remove the module in the parrallel model
            if 'module.' in list(checkpoint.items())[0][0]: 
                pretrained_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint.items())}
                checkpoint = pretrained_dict   

            # print([k for k, v in list(checkpoint.items())])
            # assert(False)
            
            # change some names in the dict
            # change_dict = {
            #     'ft_module.2':'ft_module.3',
            # }
            # for k, v in list(checkpoint.items()):
            #     for _k, _v in list(change_dict.items()):
            #         if _k in k:
            #             new_key = k.replace(_k, _v)
            #             checkpoint[new_key] = checkpoint.pop(k)
            #             break
            # change_dict = {
            #     'ft_module':'transforms',
            #     'pyramid_ext':'pyramids',
            #     'fea_bn': 'norm'
            # }
            # for k, v in list(checkpoint.items()):
            #     for _k, _v in list(change_dict.items()):
            #         if _k in k:
            #             new_key = k.replace(_k, _v)
            #             checkpoint[new_key] = checkpoint.pop(k)
            #             break

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
            print("=> Resume weigths:")
            print([k for k, v in list(pretrained_dict.items())])

            checkpoint = self.state_dict()

            unresume_dict = set(checkpoint)-set(pretrained_dict)
            print("=> UNResume weigths:")
            print(unresume_dict)

            checkpoint.update(pretrained_dict) 
            
            self.load_state_dict(checkpoint)

        else:
            print(("=> no checkpoint found at '{}'".format(resume_checkpoint)))

    def _forward_features_size(self, img_size):
        x = torch.rand(1, 3, img_size[0], img_size[1])
        x = torch.autograd.Variable(x, volatile=True).cuda()
        sources = list()
        transformed = list()
        pyramids = list()

        # apply bases layers and cache source layer outputs
        for k in range(len(self.base)):
            x = self.base[k](x)
            if k in self.feature_layer:
                sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = v(x)
            if k % 2 == 1:
                sources.append(x)
        assert len(self.transforms) == len(sources)
        upsize = (sources[0].size()[2], sources[0].size()[3])
        for k, v in enumerate(self.transforms):
            size = None if k == 0 else upsize
            transformed.append(v(sources[k], size))
        x = torch.cat(transformed,1)
        x = self.norm(x)
        for k, v in enumerate(self.pyramids):
            x = v(x)
            pyramids.append(x)
        return [(o.size()[2], o.size()[3]) for o in pyramids]


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=False, bias=True):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None
        # self.up_size = up_size
        # self.up_sample = nn.Upsample(size=(up_size,up_size),mode='bilinear') if up_size != 0 else None

    def forward(self, x, up_size=None):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        if up_size is not None:
            x = F.upsample(x, size=up_size, mode='bilinear')
            # x = self.up_sample(x)
        return x

def add_extras(base, feature_layer, mbox, num_classes):
    extra_layers = []
    feature_transform_layers = []
    pyramid_feature_layers = []
    loc_layers = []
    conf_layers = []
    in_channels = None
    feature_transform_channel = int(feature_layer[0][1][-1]/2)
    for layer, depth in zip(feature_layer[0][0], feature_layer[0][1]):
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
        feature_transform_layers += [BasicConv(in_channels, feature_transform_channel, kernel_size=1, padding=0)]
    
    in_channels = len(feature_transform_layers) * feature_transform_channel
    for layer, depth, box in zip(feature_layer[1][0], feature_layer[1][1], mbox):
        if layer == 'S':
            pyramid_feature_layers += [BasicConv(in_channels, depth, kernel_size=3, stride=2, padding=1)]
            in_channels = depth
        elif layer == '':
            pad = (0,1)[len(pyramid_feature_layers)==0]
            pyramid_feature_layers += [BasicConv(in_channels, depth, kernel_size=3, stride=1, padding=pad)]
            in_channels = depth
        else:
            AssertionError('Undefined layer')
        loc_layers += [nn.Conv2d(in_channels, box * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(in_channels, box * num_classes, kernel_size=3, padding=1)]
    return base, extra_layers, (feature_transform_layers, pyramid_feature_layers), (loc_layers, conf_layers)

def build_fssd(base, feature_layer, mbox, num_classes):
    base_, extras_, features_, head_ = add_extras(base(), feature_layer, mbox, num_classes)
    return FSSD(base_, extras_, head_, features_, feature_layer, num_classes)