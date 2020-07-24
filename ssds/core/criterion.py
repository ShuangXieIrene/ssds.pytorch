import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class MultiBoxLoss(nn.Module):
    r"""The MultiBox Loss is used to calculate the classification loss in object detection task.

    MultiBox Loss is introduce by [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325v5) and can be described as:

    .. math::
        L(x,c,l,g) = (Lconf(x, c) + \alpha Lloc(x,l,g)) / N

    where, :math:`Lconf` is the CrossEntropy Loss and :math:`Lloc` is the SmoothL1 Loss
    weighted by :math:`\alpha` which is set to 1 by cross val.

    Compute Targets:

    * Produce Confidence Target
        Indices by matching ground truth boxes
        with (default) 'priorboxes' that have jaccard index > threshold parameter
        (default threshold: 0.5).
    * Produce localization target 
        by 'encoding' variance into offsets of ground
        truth boxes and their matched  'priorboxes'.
    * Hard negative mining 
        to filter the excessive number of negative examples
        that comes with using a large number of default bounding boxes.
        (default negative:positive ratio 3:1)
    
    To reduce the code and make it more easier to embed into the pipeline. Here, only the classification loss is included in this class
    
    Args:
        negpos_ratio: ratio of negative over positive samples in the given feature map, Default: 3
    """

    def __init__(self, negpos_ratio=3, **kwargs):
        super(MultiBoxLoss, self).__init__()
        self.negpos_ratio = negpos_ratio

    def forward(self, pred_logits, target, depth):
        """
        Args:
            pred_logits: Predict class for each box
            target: Target class for each box
            depth: the sign for the positive and negative samples from anchor mathcing. \
                Basically it can be splited to 3 types: positive(>0), background/negative(=0), ignore(<0)
        Returns:
            The classification loss for the given feature map
        """
                

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
    r"""The Focal Loss is used to calculate the classification loss in object detection task.
    
    Focal Loss is introduce by [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) and can be described as:

    .. math::
        FL(p_t)=-\alpha(1-p_t)^{\gamma}ln(p_t)
    
    where :math:`p_t` is the cross entropy for each box. :math:`\alpha` controls the ratio of positive sample and the :math:`\gamma`
    controls the attention for the difficult samples.

    Args:
        alpha (float) : the param to control the ratio of positive sample, (0,1). Default: 0.25
        gamma (float) : the param to the attention for the difficult samples, [0,n), [0,5] has been shown in the original paper. Default: 2
    """

    def __init__(self, alpha=0.25, gamma=2, **kwargs):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred_logits, target, depth):
        r"""
        Args:
            pred_logits: Predict class for each box
            target: Target class for each box
            depth: Does not used in this function
        Returns:
            The classification loss for the given feature map
        """
        pred = pred_logits.sigmoid()
        ce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction="none")
        alpha = target * self.alpha + (1.0 - target) * (1.0 - self.alpha)
        pt = torch.where(target == 1, pred, 1 - pred)
        return alpha * (1.0 - pt) ** self.gamma * ce


class SmoothL1Loss(nn.Module):
    r"""The SmoothL1 Loss is used to calculate the localization loss in object detection task.
    
    This criterion that uses a squared term if the absolute
    element-wise error falls below 1 and an L1 term otherwise.
    It is less sensitive to outliers than the `MSELoss` and in some cases
    prevents exploding gradients (e.g. see `Fast R-CNN` paper by Ross Girshick).
    Also known as the Huber loss:

    .. math::
        \text{loss}(x_i, y_i) =
        \begin{cases}
        0.5 (x_i - y_i)^2, & \text{if } |x_i - y_i| < \beta \\
        |x_i - y_i| - 0.5, & \text{otherwise }
        \end{cases}
    
    :math:`x` and :math:`y` arbitrary shapes with a total of :math:`n` elements each
    the sum operation still operates over all the elements, and divides by :math:`n`.

    :math:`\beta` is used as the threshold and smooth the loss

    Args:
        beta (float) : the param to control the threshold and smooth the loss, (0,1). Default: 0.11
    """

    def __init__(self, beta=0.11):
        super().__init__()
        self.beta = beta

    def forward(self, pred, target):
        r"""
        Args:
            pred: Predict box for each box
            target: Target box for each box
        Returns:
            The localization loss for the given feature map
        """
        x = (pred - target).abs()
        l1 = x - 0.5 * self.beta
        l2 = 0.5 * x ** 2 / self.beta
        return torch.where(x >= self.beta, l1, l2)


class IOULoss(nn.Module):
    r"""The IOU Loss is used to calculate the localization loss in object detection task.

    IoU Loss is introduce by [IoU Loss for 2D/3D Object Detection](https://arxiv.org/abs/1908.03851v1) and can be described as:

    .. math::
        IoU(A, B) = \frac{A \cap B}{A \cup B} = \frac{A \cap B}{|A| + |B| - A \cap  B}
    
    where, A and B represents the two convex shapes. In here, it means the predict box and the groundtruth box.

    This class actually implemented multiple IoU related losses and use :attr:`loss_type` to choose the specific loss func.

    Args:
        loss_type (str): param to choose the specific loss type.
    """
    def __init__(self, loss_type="iou"):
        super(IOULoss, self).__init__()
        self.loss_type = loss_type

    def forward(self, pred, target):
        r"""
        Args:
            pred: Predict box for each box, format with x,y,w,h
            target: Target box for each box, format with x,y,w,h
        Returns:
            The localization loss for the given feature map
        """
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
    r"""The GIOU Loss is used to calculate the localization loss in object detection task.

    Generalized IoU Loss is introduce by [IoU Loss for 2D/3D Object Detection](https://arxiv.org/abs/1908.03851v1) and can be described as:

    .. math::
        IoU(A, B) = \frac{A \cap B}{A \cup B} = \frac{A \cap B}{|A| + |B| - A \cap  B}

    .. math::
        GIoU(A, B) = IoU(A, B) - \frac{C - U}{C}

    where, A and B represents the two convex shapes. In here, it means the predict box and the groundtruth box; C is defined as the smallest convex
    shapes enclosing both A and B; U represents the union area :math:`|A| + |B| - A \cap  B`

    In implementation, it calls the :class:`.IOULoss` with :attr:`loss_type="giou"`.
    """
    return IOULoss(loss_type="giou")


def DIOULoss():
    r"""The DIOU Loss is used to calculate the localization loss in object detection task.

    Distance IoU Loss is introduce by [Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression](https://arxiv.org/abs/1911.08287v1) and can be described as:

    .. math::
        IoU(A, B) = \frac{A \cap B}{A \cup B} = \frac{A \cap B}{|A| + |B| - A \cap  B}

    .. math::
        DIoU(A, B) = IoU(A, B) - \frac{diag_{inter}}{diag_{outer}}

    where, A and B represents the two convex shapes. In here, it means the predict box and the groundtruth box; :math:`diag_{inter}` is defined as center distance between 
    A and B; :math:`diag_{outer}` is the diagonal length of the smallest enclosing box covering the two boxes.

    In implementation, it calls the :class:`.IOULoss` with :attr:`loss_type="diou"`.
    """
    return IOULoss(loss_type="diou")


def CIOULoss():
    r"""The CIOU Loss is used to calculate the localization loss in object detection task.

    Complete IoU Loss is introduce by [Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression](https://arxiv.org/abs/1911.08287v1) and can be described as:

    .. math::
        IoU(A, B) = \frac{A \cap B}{A \cup B} = \frac{A \cap B}{|A| + |B| - A \cap  B}

    .. math::
        DIoU(A, B) = IoU(A, B) - \frac{diag_{inter}}{diag_{outer}}

    .. math::
        CIoU(A, B) = DIoU(A, B) - \alpha v
    
    where, A and B represents the two convex shapes. In here, it means the predict box and the groundtruth box; :math:`\alpha = \frac{v}{(1-IoU(A,B))+v}` 
    and :math:`v = \frac{4}{\pi^2} (arctan \frac{w^A}{h^A} − arctan \frac{w^B}{h^B})^2` is used to impose the consistency of aspect ratio.

    In CIoU loss, the :math:`\alpha` part is not used for backpropagation.

    In implementation, it calls the :class:`.IOULoss` with :attr:`loss_type="ciou"`.
    """
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
