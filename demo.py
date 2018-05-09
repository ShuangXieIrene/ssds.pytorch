from __future__ import print_function
import sys
import os
import argparse
import numpy as np
if '/data/software/opencv-3.4.0/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/data/software/opencv-3.4.0/lib/python2.7/dist-packages')
if '/data/software/opencv-3.3.1/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/data/software/opencv-3.3.1/lib/python2.7/dist-packages')
import cv2

from lib.ssds import ObjectDetector
from lib.utils.config_parse import cfg_from_file

VOC_CLASSES = ( 'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Demo a ssds.pytorch network')
    parser.add_argument('--cfg', dest='confg_file',
            help='the address of optional config file', default=None, type=str, required=True)
    parser.add_argument('--demo', dest='demo_file',
            help='the address of the demo file', default=None, type=str, required=True)
    parser.add_argument('-t', '--type', dest='type',
            help='the type of the demo file, could be "image", "video", "camera" or "time", default is "image"', default='image', type=str)
    parser.add_argument('-d', '--display', dest='display',
            help='whether display the detection result, default is True', default=True, type=bool)
    parser.add_argument('-s', '--save', dest='save',
            help='whether write the detection result, default is False', default=False, type=bool)  

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
FONT = cv2.FONT_HERSHEY_SIMPLEX

def demo(args, image_path):
    # 1. load the configure file
    cfg_from_file(args.confg_file)

    # 2. load detector based on the configure file
    object_detector = ObjectDetector()

    # 3. load image
    image = cv2.imread(image_path)

    # 4. detect
    _labels, _scores, _coords = object_detector.predict(image)

    # 5. draw bounding box on the image
    for labels, scores, coords in zip(_labels, _scores, _coords):
        cv2.rectangle(image, (int(coords[0]), int(coords[1])), (int(coords[2]), int(coords[3])), COLORS[labels % 3], 2)
        cv2.putText(image, '{label}: {score:.3f}'.format(label=VOC_CLASSES[labels], score=scores), (int(coords[0]), int(coords[1])), FONT, 0.5, COLORS[labels % 3], 2)
    
    # 6. visualize result
    if args.display is True:
        cv2.imshow('result', image)
        cv2.waitKey(0)

    # 7. write result
    if args.save is True:
        path, _ = os.path.splitext(image_path)
        cv2.imwrite(path + '_result.jpg', image)
    

def demo_live(args, video_path):
    # 1. load the configure file
    cfg_from_file(args.confg_file)

    # 2. load detector based on the configure file
    object_detector = ObjectDetector()

    # 3. load video
    video = cv2.VideoCapture(video_path)

    index = -1
    while(video.isOpened()):
        index = index + 1
        sys.stdout.write('Process image: {} \r'.format(index))
        sys.stdout.flush()

        # 4. read image
        flag, image = video.read()
        if flag == False:
            print("Can not read image in Frame : {}".format(index))
            break

        # 5. detect
        _labels, _scores, _coords = object_detector.predict(image)

        # 6. draw bounding box on the image
        for labels, scores, coords in zip(_labels, _scores, _coords):
            cv2.rectangle(image, (int(coords[0]), int(coords[1])), (int(coords[2]), int(coords[3])), COLORS[labels % 3], 2)
            cv2.putText(image, '{label}: {score:.3f}'.format(label=VOC_CLASSES[labels], score=scores), (int(coords[0]), int(coords[1])), FONT, 0.5, COLORS[labels % 3], 2)
    
        # 7. visualize result
        if args.display is True:
            cv2.imshow('result', image)
            cv2.waitKey(33)

        # 8. write result
        if args.save is True:
            path, _ = os.path.splitext(video_path)
            path = path + '_result'
            if not os.path.exists(path):
                os.mkdir(path)
            cv2.imwrite(path + '/{}.jpg'.format(index), image)        


def time_benchmark(args, image_path):
    # 1. load the configure file
    cfg_from_file(args.confg_file)

    # 2. load detector based on the configure file
    object_detector = ObjectDetector()

    # 3. load image
    image = cv2.imread(image_path)

    # 4. time test
    warmup = 20
    time_iter = 100
    print('Warmup the detector...')
    _t = list()
    for i in range(warmup+time_iter):
        _, _, _, (total_time, preprocess_time, net_forward_time, detect_time, output_time) \
            = object_detector.predict(image, check_time=True)
        if i > warmup:
            _t.append([total_time, preprocess_time, net_forward_time, detect_time, output_time])
            if i % 20 == 0: 
                print('In {}\{}, total time: {} \n preprocess: {} \n net_forward: {} \n detect: {} \n output: {}'.format(
                    i-warmup, time_iter, total_time, preprocess_time, net_forward_time, detect_time, output_time
                ))
    total_time, preprocess_time, net_forward_time, detect_time, output_time = np.sum(_t, axis=0)/time_iter * 1000 # 1000ms to 1s
    print('In average, total time: {}ms \n preprocess: {}ms \n net_forward: {}ms \n detect: {}ms \n output: {}ms'.format(
        total_time, preprocess_time, net_forward_time, detect_time, output_time
    ))
    with open('./time_benchmark.csv', 'a') as f:
        f.write("{:s},{:.2f}ms,{:.2f}ms,{:.2f}ms,{:.2f}ms,{:.2f}ms\n".format(args.confg_file, total_time, preprocess_time, net_forward_time, detect_time, output_time))


    
if __name__ == '__main__':
    args = parse_args()
    if args.type == 'image':
        demo(args, args.demo_file)
    elif args.type == 'video':
        demo_live(args, args.demo_file)
    elif args.type == 'camera':
        demo_live(args, int(args.demo_file))
    elif args.type == 'time':
        time_benchmark(args, args.demo_file)
    else:
        AssertionError('type is not correct')
