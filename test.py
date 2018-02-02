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
from lib.utils.config_parse import cfg_from_file

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a ssds.pytorch network')
    parser.add_argument('--cfg', dest='confg_file',
            help='optional config file', default=None, type=str)
    parser.add_argument('--model', dest='resume_checkpoint',
            default='./weights/SSD_vgg_VOC_epoches_270.pth',
            help='the checkpoint used to resume weight',
            type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def run_benchmark():
    args = parse_args()
    cfg_from_file(args.confg_file)

    object_detector = ObjectDetector(resume_checkpoint=args.resume_checkpoint)

    image = np.random.random((300, 300, 3))

    num_steps_burn_in = 20
    for i in range(100 + num_steps_burn_in):
        detect_times = object_detector.predict(image)
        if i >= num_steps_burn_in:
            if not i % 10:
                print('%s: step %d, detection duration = %.3f' % (datetime.now(), i - num_steps_burn_in, detect_times[1]))

if __name__ == '__main__':
    run_benchmark()
