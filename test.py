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

VOC_CLASSES = ( '__background__', # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a ssds.pytorch network')
    parser.add_argument('--cfg', dest='confg_file',
            help='optional config file', default=None, type=str)
    parser.add_argument('--model', dest='resume_checkpoint',
            default=None,
            help='the checkpoint used to resume weight',
            type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
FONT = cv2.FONT_HERSHEY_SIMPLEX

def demo():
    args = parse_args()
    cfg_from_file(args.confg_file)

    object_detector = ObjectDetector(resume_checkpoint=args.resume_checkpoint)

    image = cv2.imread('./experiments/2011_001100.jpg')
    # detect_bboxes = object_detector.predict(image)[0]
    # for class_id,class_collection in enumerate(detect_bboxes):
    #     if len(class_collection)>0:
    #         for i in range(class_collection.shape[0]):
    #             if class_collection[i,-1]>0.6:
    #                 pt = class_collection[i]
    #                 cv2.rectangle(image, (int(pt[0]), int(pt[1])), (int(pt[2]),
    #                                                                 int(pt[3])), COLORS[i % 3], 2)
    #                 cv2.putText(image, VOC_CLASSES[class_id - 1], (int(pt[0]), int(pt[1])), FONT,
    #                             0.5, (255, 255, 255), 2)
    _labels, _scores, _coords = object_detector.predict(image)
    for labels, scores, coords in zip(_labels, _scores, _coords):
        cv2.rectangle(image, (int(coords[0]), int(coords[1])), (int(coords[2]), int(coords[3])), COLORS[labels % 3], 2)
        cv2.putText(image, VOC_CLASSES[labels]+': '+str(scores), (int(coords[0]), int(coords[1])), FONT, 0.5, (255, 255, 255), 2)
    cv2.imwrite('./experiments/result.jpg',image)

if __name__ == '__main__':
    demo()
