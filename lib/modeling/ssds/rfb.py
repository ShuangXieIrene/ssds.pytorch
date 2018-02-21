import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import os

from lib.layers import *

class RFB(nn.Module):

    def __init__(self, base, extras, norm, head, feature_layer, num_classes):
        super(RFB, self).__init__()
        self.num_classes = num_classes
        # RFB network
        self.base = nn.ModuleList(base)
        self.norm = nn.ModuleList(norm)
        self.extras = nn.ModuleList(extras)
        
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        self.softmax = nn.Softmax(dim=-1)

        self.feature_layer = feature_layer[0]
        self.indicator = 0
        for layer in self.feature_layer:
            if isinstance(layer, int):
                continue
            elif layer == '' or layer == 'S':
                break
            else:
                self.indicator += 1 



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
                idx = self.feature_layer.index(k)
                # some xxx
                if(len(sources)) == 0:
                    sources.append(self.norm[idx](x))
                else:
                    x = self.norm[idx](x)
                    sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = v(x)
            if k < self.indicator or k % 2 == 1:
                sources.append(x)

        # print([o.size() for o in sources])
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
    
    def _forward_features_size(self, img_size):
        x = torch.rand(1, 3, img_size[0], img_size[1])
        x = torch.autograd.Variable(x, volatile=True).cuda()
        sources = list()
        self.eval()
        # apply bases layers and cache source layer outputs
        for k in range(len(self.base)):
            x = self.base[k](x)
            if k in self.feature_layer:
                idx = self.feature_layer.index(k)
                sources.append(self.norm[idx](x))

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = v(x)
            if k < self.indicator or k % 2 == 1:
                sources.append(x)

        return [(o.size()[2], o.size()[3]) for o in sources]

    def load_weights(self, resume_checkpoint, resume_scope=''):
        if os.path.isfile(resume_checkpoint):
            print(("=> loading checkpoint '{}'".format(resume_checkpoint)))
            checkpoint = torch.load(resume_checkpoint)

            # remove the module in the parrallel model
            if 'module.' in list(checkpoint.items())[0][0]: 
                pretrained_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint.items())}
                checkpoint = pretrained_dict   

            # change some names in the dict
            # change_dict = {
            #     'Norm':'norm.0',
            #     'extras.0':'norm.1',
            #     'extras.1':'extras.0',
            #     'extras.2':'extras.1',
            #     'extras.3':'extras.2',
            #     'extras.4':'extras.3',
            #     'extras.5':'extras.4',
            #     'extras.6':'extras.5',
            # }
            # for k, v in list(checkpoint.items()):
            #     for _k, _v in list(change_dict.items()):
            #         if _k in k:
            #             new_key = k.replace(_k, _v)
            #             checkpoint[new_key] = checkpoint.pop(k)
            # print([k for k, v in list(checkpoint.items())])
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

            print("=> Weigths in the checkpoints:")
            print([k for k, v in list(checkpoint.items())])

            pretrained_dict = {k: v for k, v in checkpoint.items() if k in self.state_dict()}
            checkpoint = self.state_dict()
            checkpoint.update(pretrained_dict) 
            
            print("=> Resume weigths:")
            print([k for k, v in list(pretrained_dict.items())])

            self.load_state_dict(checkpoint)

        else:
            print(("=> no checkpoint found at '{}'".format(resume_checkpoint)))


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None
    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class BasicSepConv(nn.Module):
    def __init__(self, in_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicSepConv, self).__init__()
        self.out_channels = in_planes
        self.conv = nn.Conv2d(in_planes, in_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(in_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None
    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class BasicRFB_a(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale = 0.1):
        super(BasicRFB_a, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes //4

        self.branch0 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=1,relu=False)
                )
        self.branch1 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, inter_planes, kernel_size=(3,1), stride=1, padding=(1,0)),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
                )
        self.branch2 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, inter_planes, kernel_size=(1,3), stride=stride, padding=(0,1)),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
                )
        self.branch3 = nn.Sequential(
                BasicConv(in_planes, inter_planes//2, kernel_size=1, stride=1),
                BasicConv(inter_planes//2, (inter_planes//4)*3, kernel_size=(1,3), stride=1, padding=(0,1)),
                BasicConv((inter_planes//4)*3, inter_planes, kernel_size=(3,1), stride=stride, padding=(1,0)),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
                )

        self.ConvLinear = BasicConv(4*inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self,x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        out = torch.cat((x0,x1,x2,x3),1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out*self.scale + short
        out = self.relu(out)
        return out

class BasicRFB(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, scale = 0.1, visual = 1):
        super(BasicRFB, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 8
        self.branch0 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                #BasicConv(inter_planes, (inter_planes//2)*3, kernel_size=(1,3), stride=1, padding=(0,1)),
                BasicConv(inter_planes, 2*inter_planes, kernel_size=(3,3), stride=stride, padding=(1,1)),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=visual+1, dilation=visual+1, relu=False)
                )
        self.branch1 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, (inter_planes//2)*3, kernel_size=3, stride=1, padding=1),
                BasicConv((inter_planes//2)*3, 2*inter_planes, kernel_size=3, stride=stride, padding=1),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=2*visual+1, dilation=2*visual+1, relu=False)
                )

        self.ConvLinear = BasicConv(4*inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self,x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)

        out = torch.cat((x0,x1),1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out*self.scale + short
        out = self.relu(out)

        return out

def add_extras(base, feature_layer, mbox, num_classes):
    extra_layers = []
    loc_layers = []
    conf_layers = []
    norm_layers = []
    in_channels = None
    for layer, depth, box in zip(feature_layer[0], feature_layer[1], mbox):
        if layer == 'RBF':
            extra_layers += [BasicRFB(in_channels, depth, stride=2, scale = 1.0, visual=2)]
            in_channels = depth
        elif layer == 'S':
            extra_layers += [
                    BasicConv(in_channels, int(depth/2), kernel_size=1),
                    BasicConv(int(depth/2), depth, kernel_size=3, stride=2, padding=1)  ]
            in_channels = depth
        elif layer == '':
            extra_layers += [
                    BasicConv(in_channels, int(depth/2), kernel_size=1),
                    BasicConv(int(depth/2), depth, kernel_size=3)  ]
            in_channels = depth
        else:
            if len(norm_layers) == 0:
                norm_layers += [BasicRFB_a(depth, depth, stride = 1, scale=1.0)]
            else:
                norm_layers += [BasicRFB(depth, depth, scale = 1.0, visual=2)]
            in_channels = depth
        loc_layers += [nn.Conv2d(in_channels, box * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(in_channels, box * num_classes, kernel_size=3, padding=1)]
    return base, extra_layers, norm_layers, (loc_layers, conf_layers)

def build_rfb(base, feature_layer, mbox, num_classes):
    base_, extras_, norm_, head_ = add_extras(base(), feature_layer, mbox, num_classes)
    return RFB(base_, extras_, norm_, head_, feature_layer, num_classes)