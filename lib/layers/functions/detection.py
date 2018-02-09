import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Function
from torch.autograd import Variable
from lib.utils.box_utils import decode, nms


class Detect(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """
    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh, variance=[0.1, 0.2]):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.variance = variance
        self.output = torch.zeros(1, self.num_classes, self.top_k, 5)

    # def forward(self, predictions, prior):
    #     """
    #     Args:
    #         loc_data: (tensor) Loc preds from loc layers
    #             Shape: [batch,num_priors*4]
    #         conf_data: (tensor) Shape: Conf preds from conf layers
    #             Shape: [batch*num_priors,num_classes]
    #         prior_data: (tensor) Prior boxes and variances from priorbox layers
    #             Shape: [1,num_priors,4]
    #     """
    #     loc, conf = predictions

    #     loc_data = loc.data
    #     conf_data = conf.data
    #     prior_data = prior.data

    #     num = loc_data.size(0)  # batch size
    #     num_priors = prior_data.size(0)
    #     self.boxes = torch.zeros(1, num_priors, 4)
    #     self.scores = torch.zeros(1, num_priors, self.num_classes)

    #     if num == 1:
    #         # size batch x num_classes x num_priors
    #         conf_preds = conf_data.unsqueeze(0)

    #     else:
    #         conf_preds = conf_data.view(num, num_priors,
    #                                     self.num_classes)
    #         self.boxes.expand_(num, num_priors, 4)
    #         self.scores.expand_(num, num_priors, self.num_classes)

    #     # Decode predictions into bboxes.
    #     for i in range(num):
    #         decoded_boxes = decode(loc_data[i], prior_data, self.variance)
    #         # For each class, perform nms
    #         conf_scores = conf_preds[i].clone()
    #         '''
    #         c_mask = conf_scores.gt(self.thresh)
    #         decoded_boxes = decoded_boxes[c_mask]
    #         conf_scores = conf_scores[c_mask]
    #         '''

    #         self.boxes[i] = decoded_boxes
    #         self.scores[i] = conf_scores

    #     return self.boxes, self.scores


    def forward(self, predictions, prior):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """
        loc, conf = predictions

        loc_data = loc.data
        conf_data = conf.data
        prior_data = prior.data

        num = loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)
        self.output.zero_()
        if num == 1:
            # size batch x num_classes x num_priors
            conf_preds = conf_data.t().contiguous().unsqueeze(0)
        else:
            conf_preds = conf_data.view(num, num_priors,
                                        self.num_classes).transpose(2, 1)
            self.output.expand_(num, self.num_classes, self.top_k, 5)

        # Decode predictions into bboxes.
        for i in range(num):
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()
            num_det = 0
            for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                if scores.dim() == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # idx of highest scoring and non-overlapping boxes per class
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                self.output[i, cl, :count] = \
                    torch.cat((scores[ids[:count]].unsqueeze(1),
                               boxes[ids[:count]]), 1)
        flt = self.output.view(-1, 5)
        _, idx = flt[:, 0].sort(0)
        _, rank = idx.sort(0)
        flt[(rank >= self.top_k).unsqueeze(1).expand_as(flt)].fill_(0)
        return self.output