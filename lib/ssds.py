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
from lib.utils.config_parse import cfg

class ObjectDetector:
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

        self.net = net
        if prior_box is None:
            prior_box = cfg.MODEL.PRIOR_BOX
        self.priorbox = PriorBox(prior_box)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.detection = Detect(num_classes, 0, 200, 0.01, 0.45, self.priorbox.variance)
        self.transform = BaseTransform(300, (104, 117, 123), (2,0,1))
        self.max_per_image = 300
        self.num_classes = num_classes
        self.max_per_image = max_per_image
        self.cuda = cuda
        self.thresh = thresh


    def predict(self,img):
        scale = torch.Tensor([img.shape[1], img.shape[0],
                              img.shape[1], img.shape[0]]).cpu().numpy()
        _t = {'im_detect': Timer(), 'misc': Timer()}
        assert img.shape[2] == 3
        x = Variable(self.transform(img).unsqueeze(0),volatile=True)
        if self.cuda:
            x = x.cuda()
        _t['im_detect'].tic()
        out = self.net(x)  # forward pass
        boxes, scores = self.detection.forward(out, self.priors)
        detect_time = _t['im_detect'].toc()
        boxes = boxes[0]
        scores = scores[0]

        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()
        # scale each detection back up to the image
        boxes *= scale
        _t['misc'].tic()
        all_boxes = [[] for _ in range(self.num_classes)]

        for j in range(1, self.num_classes):
            inds = np.where(scores[:, j] > self.thresh)[0]
            if len(inds) == 0:
                all_boxes[j] = np.zeros([0, 5], dtype=np.float32)
                continue
            c_bboxes = boxes[inds]
            c_scores = scores[inds, j]
            print(scores[:,j])
            c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
                np.float32, copy=False)
            keep = nms(c_dets, 0.45)
            # keep = py_cpu_nms(c_dets, 0.45)
            keep = keep[:50]
            c_dets = c_dets[keep, :]
            all_boxes[j] = c_dets
        if self.max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][:, -1] for j in range(1, self.num_classes)])
            if len(image_scores) > self.max_per_image:
                image_thresh = np.sort(image_scores)[-self.max_per_image]
                for j in range(1, self.num_classes):
                    keep = np.where(all_boxes[j][:, -1] >= image_thresh)[0]
                    all_boxes[j] = all_boxes[j][keep, :]

        nms_time = _t['misc'].toc()

        return all_boxes, detect_time, nms_time