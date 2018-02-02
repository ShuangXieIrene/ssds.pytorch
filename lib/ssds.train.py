from __future__ import print_function
import numpy as np

import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from lib.layers import *
from lib.utils.timer import Timer
from lib.utils.nms_wrapper import nms
from lib.utils.data_augment import BaseTransform

from lib.nets import net_factory
from lib.models import model_factory


class SolverWrapper(object):
    """
    A wrapper class for the training process
    """
    def __init__(self, args, resume_checkpoint=None, num_classes = 21, cuda = False):

        base_fn = net_factory.gen_base_fn(name=args.base_fn)
        net = model_factory.gen_model_fn(name=args.model_fn)(base = base_fn)
        if resume_checkpoint:
            net.load_weights(resume_checkpoint)
        
        if cuda:
            model = model.cuda()
            model = torch.nn.DataParallel(model, device_ids=[int(i) for i in args.gpus.strip().split(',')])
            cudnn.benchmark = True

        self.net = net
        self.priorbox = PriorBox(VOC_300)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.detection = Detect(num_classes, 0, 200, 0.01, 0.45, self.priorbox.variance)
        self.transform = BaseTransform(300, (104, 117, 123), (2,0,1))
        self.max_per_image = 300
        self.num_classes = num_classes
        self.cuda = cuda


    def predict(self,img):

    def __init__(self, base_fn=None, model_fn=None, transform=None, prior_box=None,
                resume_checkpoint=None, num_classes = 21, cuda = True, 
                max_per_image = 300, thresh = 0.5):
        if base_fn is None:
            base_fn = cfg.MODEL.BASE_FN
        base = net_factory.gen_base_fn(name=base_fn)
        if model_fn is None:
            model_fn = cfg.MODEL.MODEL_FN
        net = model_factory.gen_model_fn(name=model_fn)(base=base)
        if resume_checkpoint:
            net.load_weights(resume_checkpoint)
        net.eval()
        if cuda:
            net = net.cuda()
            cudnn.benchmark = True

def train_model(base_fn=None, model_fn=None, resume_checkpoint=None, cuda = True):
    if base_fn is None:
        base_fn = cfg.MODEL.BASE_FN
    base = net_factory.gen_base_fn(name=base_fn)
    if model_fn is None:
        model_fn = cfg.MODEL.MODEL_FN
    net = model_factory.gen_model_fn(name=model_fn)(base=base)
    if resume_checkpoint:
        net.load_weights(resume_checkpoint)
    if cuda:
        net = net.cuda()
        cudnn.benchmark = True

    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = MultiBoxLoss(num_classes, 0.5, True, 0, True, 3, 0.5, False, args.cuda)
    return True