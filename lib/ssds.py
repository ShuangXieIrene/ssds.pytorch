from __future__ import print_function
import numpy as np

import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from lib.layers import *
from lib.utils.timer import Timer
from lib.utils.nms_wrapper import nms
from lib.dataset.data_augment import BaseTransform

from lib.nets import net_factory
from lib.models import model_factory
from lib.utils.config_parse import cfg


# def py_cpu_nms(dets, thresh):
#     """Pure Python NMS baseline."""
#     x1 = dets[:, 0]
#     y1 = dets[:, 1]
#     x2 = dets[:, 2]
#     y2 = dets[:, 3]
#     scores = dets[:, 4]

#     areas = (x2 - x1 + 1) * (y2 - y1 + 1)
#     order = scores.argsort()[::-1]

#     keep = []
#     while order.size > 0:
#         i = order[0]
#         keep.append(i)
#         xx1 = np.maximum(x1[i], x1[order[1:]])
#         yy1 = np.maximum(y1[i], y1[order[1:]])
#         xx2 = np.minimum(x2[i], x2[order[1:]])
#         yy2 = np.minimum(y2[i], y2[order[1:]])

#         w = np.maximum(0.0, xx2 - xx1 + 1)
#         h = np.maximum(0.0, yy2 - yy1 + 1)
#         inter = w * h
#         ovr = inter / (areas[i] + areas[order[1:]] - inter)

#         inds = np.where(ovr <= thresh)[0]
#         order = order[inds + 1]

#     return keep

class ObjectDetector:
    def __init__(self, base_fn=None, model_fn=None, transform=None, 
                resume_checkpoint=None, num_classes = 21, cuda = True, 
                max_per_image = 300, thresh = 0.5):
        if base_fn is None:
            base_fn = cfg.MODEL.BASE_FN
        base = net_factory.gen_base_fn(name=base_fn)
        if model_fn is None:
            model_fn = cfg.MODEL.MODEL_FN
        self.net = model_factory.gen_model_fn(name=model_fn)(base=base, 
                    feature_layer=cfg.MODEL.FEATURE_LAYER, layer_depth=cfg.MODEL.LAYER_DEPTH, mbox=cfg.MODEL.MBOX, num_classes=21)
        if resume_checkpoint is None:
            resume_checkpoint=cfg.RESUME_CHECKPOINT
        if resume_checkpoint != '':
            self.net.load_weights(resume_checkpoint)
        self.net.eval()
        if cuda:
            self.net = self.net.cuda()
            cudnn.benchmark = True

        feature_maps = self.net._forward_features_size(cfg.MODEL.IMAGE_SIZE)
        priorbox = PriorBox(image_size=cfg.MODEL.IMAGE_SIZE, feature_maps=feature_maps, aspect_ratios=cfg.MODEL.PRIOR_BOX.ASPECT_RATIOS, 
                    scale=cfg.MODEL.PRIOR_BOX.SIZES, archor_stride=cfg.MODEL.PRIOR_BOX.STEPS, clip=cfg.MODEL.PRIOR_BOX.CLIP)
        self.priors = Variable(priorbox.forward(), volatile=True)

        self.detection = Detect(num_classes, cfg.MODEL.POST_PROCESS.NUM_CLASSES, cfg.MODEL.POST_PROCESS.MAX_DETECTIONS, 
                                cfg.MODEL.POST_PROCESS.SCORE_THRESHOLD, cfg.MODEL.POST_PROCESS.IOU_THRESHOLD)
        self.transform = BaseTransform(300, (104, 117, 123), (2,0,1))
        self.num_classes = num_classes
        self.max_per_image = max_per_image
        self.cuda = cuda
        self.thresh = thresh


    # def predict(self,img):
    #     scale = torch.Tensor([img.shape[1], img.shape[0],
    #                           img.shape[1], img.shape[0]]).cpu().numpy()
    #     _t = {'im_detect': Timer(), 'misc': Timer()}
    #     assert img.shape[2] == 3
    #     x = Variable(self.transform(img).unsqueeze(0),volatile=True)
    #     if self.cuda:
    #         x = x.cuda()
    #     _t['im_detect'].tic()
    #     out = self.net(x)  # forward pass
    #     boxes, scores = self.detection.forward(out, self.priors)
    #     detect_time = _t['im_detect'].toc()
    #     boxes = boxes[0]
    #     scores = scores[0]

    #     boxes = boxes.cpu().numpy()
    #     scores = scores.cpu().numpy()
    #     # scale each detection back up to the image
    #     boxes *= scale
    #     _t['misc'].tic()
    #     all_boxes = [[] for _ in range(self.num_classes)]

    #     for j in range(1, self.num_classes):
    #         inds = np.where(scores[:, j] > self.thresh)[0]
    #         if len(inds) == 0:
    #             all_boxes[j] = np.zeros([0, 5], dtype=np.float32)
    #             continue
    #         c_bboxes = boxes[inds]
    #         c_scores = scores[inds, j]
    #         # print(scores[:,j])
    #         c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
    #             np.float32, copy=False)
    #         # keep = nms(c_dets, 0.45)
    #         keep = py_cpu_nms(c_dets, 0.45)
    #         keep = keep[:50]
    #         c_dets = c_dets[keep, :]
    #         all_boxes[j] = c_dets
    #     if self.max_per_image > 0:
    #         image_scores = np.hstack([all_boxes[j][:, -1] for j in range(1, self.num_classes)])
    #         if len(image_scores) > self.max_per_image:
    #             image_thresh = np.sort(image_scores)[-self.max_per_image]
    #             for j in range(1, self.num_classes):
    #                 keep = np.where(all_boxes[j][:, -1] >= image_thresh)[0]
    #                 all_boxes[j] = all_boxes[j][keep, :]

    #     nms_time = _t['misc'].toc()

    #     return all_boxes, detect_time, nms_time

    def predict(self,img):
        # scale = torch.Tensor([img.shape[1], img.shape[0],
        #                       img.shape[1], img.shape[0]]).cpu().numpy()
        
        _t = {'im_detect': Timer(), 'misc': Timer()}
        assert img.shape[2] == 3
        x = Variable(self.transform(img).unsqueeze(0),volatile=True)
        if self.cuda:
            x = x.cuda()
        _t['im_detect'].tic()
        out = self.net(x)  # forward pass
        detections = self.detection.forward(out, self.priors)
        detect_time = _t['im_detect'].toc()

        scale = torch.Tensor([img.shape[1::-1], img.shape[1::-1]])
        # print(scale)
        labels = []
        scores = []
        coords = []
        # for batch in range(detections.size(0)):
        #     print('Batch:', batch)
        batch=0
        for classes in range(detections.size(1)):
            num = 0
            # print('classes:', classes
            while detections[batch,classes,num,0] >= 0.6:
                scores.append(detections[batch,classes,num,0])
                labels.append(classes-1)
                print('coords:')
                coords.append(detections[batch,classes,num,1:]*scale)
                num+=1

        # print(scores)
        # print(labels)
        # print(coords)
        # assert(False)
        #return detections #, detect_time, nms_time
        return labels, scores, coords