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
from lib.ssds.train import train_model



def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--cfg', dest='cfg_file',
            help='optional config file', default=None, type=str)
    parser.add_argument('--model', dest='model',
            help='model to test',
            default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
            help='dataset to test',
            default='voc_2007_test', type=str)
    parser.add_argument('--comp', dest='comp_mode', help='competition mode',
            action='store_true')
    parser.add_argument('--num_dets', dest='max_per_image',
            help='max number of detections per image',
            default=100, type=int)
    parser.add_argument('--tag', dest='tag',
                        help='tag of the model',
                        default='', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152, mobile',
                        default='res50', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args
parser = argparse.ArgumentParser(description='ssds.pytorch')
parser.add_argument('--trained_model', default='./weights/SSD_vgg_VOC_epoches_270.pth',
                    type=str, help='Trained state_dict file path to open')
args = parser.parse_args()



def train_model():
    object_detector = ObjectDetector(cuda = args.cuda, resume_checkpoint=args.trained_model)

    image = np.random.random((300, 300, 3))

    num_steps_burn_in = 20
    for i in range(100 + num_steps_burn_in):
        detect_times = object_detector.predict(image)
        if i >= num_steps_burn_in:
            if not i % 10:
                print('%s: step %d, detection duration = %.3f' % (datetime.now(), i - num_steps_burn_in, detect_times[1]))

if __name__ == '__main__':
    run_benchmark()
