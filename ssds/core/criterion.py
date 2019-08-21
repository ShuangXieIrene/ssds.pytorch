import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from ssds.utils.box_utils import match, log_sum_exp


class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, cfg, device):
        super(MultiBoxLoss, self).__init__()
        self.device = device
        self.num_classes = cfg.NUM_CLASSES
        self.background_label = cfg.BACKGROUND_LABEL
        self.negpos_ratio = cfg.NEGPOS_RATIO
        self.threshold = cfg.MATCHED_THRESHOLD
        self.unmatched_threshold = cfg.UNMATCHED_THRESHOLD
        self.variance = cfg.VARIANCE
        # self.priors = priors

    def forward(self, predictions, targets, priors):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)
            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        loc_data, conf_data = predictions
        num = loc_data.size(0)

        num_priors = priors.size(1)
        num_classes = self.num_classes

        # print(loc_data.shape, conf_data.shape, priors.shape, len(targets))

        # assert False

        # match priors (default boxes) and ground truth boxes
        loc_t = priors.new(num, 4, num_priors).float() #torch.Tensor(num, num_priors, 4)
        conf_t = priors.new(num, num_priors).long() #torch.LongTensor(num, num_priors)
        for idx in range(num):
            truths = targets[idx][:-1].data
            labels = targets[idx][-1].data
            defaults = priors.data
            loc_t[idx], conf_t[idx] = match(truths, labels,
                                            defaults, self.variance,
                                            self.threshold)
        
        # TODO: I may did a wrong thing, may be should change the order at func input..
        loc_t = loc_t.permute(0,2,1).contiguous()
        loc_data = loc_data.permute(0,2,1).contiguous()
        conf_data = conf_data.permute(0,2,1).contiguous()

        pos = conf_t > 0
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_t)
        loc_p = loc_data[pos_idx].view(-1,4)
        loc_t = loc_t[pos_idx].view(-1,4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1,1))

        # Hard Negative Mining
        loss_c[pos.view(-1, 1)] = 0 # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)
        _,loss_idx = loss_c.sort(1, descending=True)
        _,idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1,keepdim=True) #new sum needs to keep the same dim
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1,self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N = num_pos.data.sum()
        loss_l/=N
        loss_c/=N
        return loss_l,loss_c


def configure_criterion(criterion):
    if criterion == 'MultiBoxLoss':
        c = MultiBoxLoss
    else:
        AssertionError('criterion can not be recognized')
    return c