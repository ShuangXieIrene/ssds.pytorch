from __future__ import print_function
import sys
import os
import argparse
import numpy as np
sys.path.remove('/data/software/opencv-3.4.0/lib/python2.7/dist-packages')
import cv2
from datetime import datetime

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable

from lib.models.ssd import build_ssd
from lib.ssds import ObjectDetector

parser = argparse.ArgumentParser(description='ssds.pytorch')
parser.add_argument('--trained_model', default='./weights/SSD_vgg_VOC_epoches_270.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')
args = parser.parse_args()


VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

def run_benchmark():
    net = build_ssd(net = 'vgg16')
    state_dict = torch.load(args.trained_model)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():        
        head = k[:7]
        if head == 'module.':
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)
    net.eval()

    # load data
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True

    object_detector = ObjectDetector(net, cuda = args.cuda)

    image = np.random.random((300, 300, 3))

    num_steps_burn_in = 20
    for i in range(100 + num_steps_burn_in):
        detect_times = object_detector.predict(image)
        if i >= num_steps_burn_in:
            if not i % 10:
                print('%s: step %d, detection duration = %.3f' % (datetime.now(), i - num_steps_burn_in, detect_times[1]))

if __name__ == '__main__':
    run_benchmark()
