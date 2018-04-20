from __future__ import print_function
import numpy as np

import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from lib.layers import *
from lib.utils.timer import Timer
from lib.utils.data_augment import preproc
from lib.modeling.model_builder import create_model
from lib.utils.config_parse import cfg

class ObjectDetector:
    def __init__(self, viz_arch=False):
        self.cfg = cfg

        # Build model
        print('===> Building model')
        self.model, self.priorbox = create_model(cfg.MODEL)
        self.priors = Variable(self.priorbox.forward(), volatile=True)

        # Print the model architecture and parameters
        if viz_arch is True:
            print('Model architectures:\n{}\n'.format(self.model))

        # Utilize GPUs for computation
        self.use_gpu = torch.cuda.is_available()
        self.half = False
        if self.use_gpu:
            print('Utilize GPUs for computation')
            print('Number of GPU available', torch.cuda.device_count())
            self.model.cuda()
            self.priors.cuda()
            cudnn.benchmark = True
            # self.model = torch.nn.DataParallel(self.model).module
            # Utilize half precision
            self.half = cfg.MODEL.HALF_PRECISION
            if self.half:
                self.model = self.model.half()
                self.priors = self.priors.half()
        
        # Build preprocessor and detector
        self.preprocessor = preproc(cfg.MODEL.IMAGE_SIZE, cfg.DATASET.PIXEL_MEANS, -2)
        self.detector = Detect(cfg.POST_PROCESS, self.priors)

        # Load weight:
        if cfg.RESUME_CHECKPOINT == '':
            AssertionError('RESUME_CHECKPOINT can not be empty')
        print('=> loading checkpoint {:s}'.format(cfg.RESUME_CHECKPOINT))
        checkpoint = torch.load(cfg.RESUME_CHECKPOINT)
        # checkpoint = torch.load(cfg.RESUME_CHECKPOINT, map_location='gpu' if self.use_gpu else 'cpu')
        self.model.load_state_dict(checkpoint)

        # test only
        self.model.eval()


    def predict(self, img, threshold=0.6, check_time=False):
        # make sure the input channel is 3 
        assert img.shape[2] == 3
        scale = torch.Tensor([img.shape[1::-1], img.shape[1::-1]])
        
        _t = {'preprocess': Timer(), 'net_forward': Timer(), 'detect': Timer(), 'output': Timer()}
        
        # preprocess image
        _t['preprocess'].tic()
        x = Variable(self.preprocessor(img)[0].unsqueeze(0),volatile=True)
        if self.use_gpu:
            x = x.cuda()
        if self.half:
            x = x.half()
        preprocess_time = _t['preprocess'].toc()

        # forward
        _t['net_forward'].tic()
        out = self.model(x)  # forward pass
        net_forward_time = _t['net_forward'].toc()

        # detect
        _t['detect'].tic()
        detections = self.detector.forward(out)
        detect_time = _t['detect'].toc()
        
        # output
        _t['output'].tic()
        labels, scores, coords = [list() for _ in range(3)]
        # for batch in range(detections.size(0)):
        #     print('Batch:', batch)
        batch=0
        for classes in range(detections.size(1)):
            num = 0
            while detections[batch,classes,num,0] >= threshold:
                scores.append(detections[batch,classes,num,0])
                labels.append(classes-1)
                coords.append(detections[batch,classes,num,1:]*scale)
                num+=1
        output_time = _t['output'].toc()
        total_time = preprocess_time + net_forward_time + detect_time + output_time
        
        if check_time is True:
            return labels, scores, coords, (total_time, preprocess_time, net_forward_time, detect_time, output_time)
            # total_time = preprocess_time + net_forward_time + detect_time + output_time
            # print('total time: {} \n preprocess: {} \n net_forward: {} \n detect: {} \n output: {}'.format(
            #     total_time, preprocess_time, net_forward_time, detect_time, output_time
            # ))
        return labels, scores, coords