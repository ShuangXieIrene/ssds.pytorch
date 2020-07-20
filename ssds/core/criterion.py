import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class MultiBoxLoss(nn.Module):
    r"""SSD Weighted Loss Function
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
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        c: class confidences,
        l: predicted boxes,
        g: ground truth boxes
        N: number of matched default boxes
    """

    def __init__(self, negpos_ratio=3, **kwargs):
        super(MultiBoxLoss, self).__init__()
        self.negpos_ratio = negpos_ratio

    def forward(self, pred_logits, target, depth):
        """Multibox Loss
        Args:
            depth (): 
        """
                # > 0: positive
                # = 0: background
                # < 0: ignore

        pred = pred_logits.sigmoid()
        ce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction="none")

        # Hard Negative Mining
        max_ce = ce.max(2)[0].view(ce.shape[0], -1)
        depth_v = depth.view(ce.shape[0], -1)
        max_ce[depth_v != 0] = 0  # include the pos and ignore
        _, idx = max_ce.sort(1, descending=True)
        _, idx_rank = idx.sort(1)

        # select top n neg
        num_pos = (depth_v > 0).sum((1))
        num_neg = torch.clamp(self.negpos_ratio * num_pos, max=depth_v.shape[1] - 1)
        neg = idx_rank < num_neg.expand_as(idx_rank)
        neg = neg.view_as(depth)

        return ce * ((depth > 0) + neg).gt(0).expand_as(ce)


class FocalLoss(nn.Module):
    r"Focal Loss - https://arxiv.org/abs/1708.02002"

    def __init__(self, alpha=0.25, gamma=2, **kwargs):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred_logits, target, depth):
        r"""
        Args:
            pred_logits:
            target:
            depth:
        Returns:
            The classification loss
        """
        pred = pred_logits.sigmoid()
        ce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction="none")
        alpha = target * self.alpha + (1.0 - target) * (1.0 - self.alpha)
        pt = torch.where(target == 1, pred, 1 - pred)
        return alpha * (1.0 - pt) ** self.gamma * ce


class SmoothL1Loss(nn.Module):
    "Smooth L1 Loss"

    def __init__(self, beta=0.11):
        super().__init__()
        self.beta = beta

    def forward(self, pred, target):
        x = (pred - target).abs()
        l1 = x - 0.5 * self.beta
        l2 = 0.5 * x ** 2 / self.beta
        return torch.where(x >= self.beta, l1, l2)


class IOULoss(nn.Module):
    def __init__(self, loss_type="iou"):
        super(IOULoss, self).__init__()
        self.loss_type = loss_type

    def forward(self, pred, target):
        pred_lt, pred_rb, pred_wh = self.delta2ltrb(pred)
        target_lt, target_rb, target_wh = self.delta2ltrb(target)

        lt = torch.max(pred_lt, target_lt)
        rb = torch.min(pred_rb, target_rb)

        area_i = torch.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
        area_a = torch.prod(pred_wh, axis=2)
        area_b = torch.prod(target_wh, axis=2)

        area_union = area_a + area_b - area_i
        iou = (area_i + 1e-7) / (area_union + 1e-7)

        if self.loss_type == "iou":
            iou = torch.clamp(iou, min=0, max=1.0).unsqueeze(2)
            return 1 - iou

        outer_lt = torch.min(pred_lt, target_lt)
        outer_rb = torch.max(pred_rb, target_rb)

        if self.loss_type == "giou":
            area_outer = (
                torch.prod(outer_rb - outer_lt, axis=2)
                * (outer_lt < outer_rb).all(axis=2)
                + 1e-7
            )
            giou = iou - (area_outer - area_union) / area_outer
            giou = torch.clamp(giou, min=-1.0, max=1.0).unsqueeze(2)
            return 1 - giou

        inter_diag = ((pred[:, :, :2] - target[:, :, :2]) ** 2).sum(dim=2)
        outer_diag = ((outer_rb - outer_lt) ** 2).sum(dim=2) + 1e-7

        if self.loss_type == "diou":
            diou = iou - inter_diag / outer_diag
            diou = torch.clamp(diou, min=-1.0, max=1.0).unsqueeze(2)
            return 1 - diou

        if self.loss_type == "ciou":
            v = (4 / (math.pi ** 2)) * torch.pow(
                (
                    torch.atan(target_wh[:, :, 0] / target_wh[:, :, 1])
                    - torch.atan(pred_wh[:, :, 0] / pred_wh[:, :, 1])
                ),
                2,
            )
            with torch.no_grad():
                S = 1 - iou
                alpha = v / (S + v)
            ciou = iou - (inter_diag / outer_diag + alpha * v)
            ciou = torch.clamp(ciou, min=-1.0, max=1.0).unsqueeze(2)
            return 1 - ciou

    def delta2ltrb(self, deltas):
        """ deltas [x,y,w,h] with [batch, anchor, 4, h, w]
        """
        pred_ctr = deltas[:, :, :2]
        pred_wh = torch.exp(deltas[:, :, 2:])
        return pred_ctr - 0.5 * pred_wh, pred_ctr + 0.5 * pred_wh, pred_wh


def GIOULoss():
    return IOULoss(loss_type="giou")


def DIOULoss():
    return IOULoss(loss_type="diou")


def CIOULoss():
    return IOULoss(loss_type="ciou")


if __name__ == "__main__":
    iou = IOULoss()
    giou = GIOULoss()
    diou = DIOULoss()

    box = torch.tensor([[[0.0, 0.0, 0.5, 0.5]]])
    box[:, :, 2:] = torch.log(box[:, :, 2:])
    tar = torch.tensor([[[0, 0, 1.0, 1.0]]])
    tar[:, :, 2:] = torch.log(tar[:, :, 2:])

    print("IOU: ", iou(box, tar))
    print("GIOU: ", giou(box, tar))
    print("DIOU: ", diou(box, tar))
