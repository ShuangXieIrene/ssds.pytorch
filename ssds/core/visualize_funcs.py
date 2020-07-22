import torch
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment


def matrix_iou(a, b):
    """
    return iou of a and b, numpy version for data augenmentation

    :meta private:
    """
    lt = np.maximum(a[:, None, :2], b[:, :2])
    rb = np.minimum(a[:, None, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.nan_to_num(area_i / (area_a[:, None] + area_b - area_i))


def add_prCurve(writer, precision, recall, class_names=[], epoch=0):
    def add_pr_curve_raw(writer, tag, precision, recall, epoch=0):
        """ the pr_curve_raw_data_pb() needs
        Args:
            precisions: ascending  array
            recalls   : descending array
        """
        num_thresholds = len(precision)
        writer.add_pr_curve_raw(
            tag=tag,
            true_positive_counts=-np.ones(num_thresholds),
            false_positive_counts=-np.ones(num_thresholds),
            true_negative_counts=-np.ones(num_thresholds),
            false_negative_counts=-np.ones(num_thresholds),
            precision=precision,
            recall=recall,
            global_step=epoch,
            num_thresholds=num_thresholds,
        )

    for i, (_prec, _rec) in enumerate(zip(precision, recall)):
        num_thresholds = min(500, len(_prec))
        if num_thresholds != len(_prec):
            gap = int(len(_prec) / num_thresholds)
            _prec = np.append(_prec[::gap], _prec[-1])
            _rec = np.append(_rec[::gap], _rec[-1])
            num_thresholds = len(_prec)
        _prec.sort()
        _rec[::-1].sort()
        tag = class_names[i] if class_names else "pr_curve/{}".format(i + 1)
        add_pr_curve_raw(
            writer=writer, tag=tag, precision=_prec, recall=_rec, epoch=epoch
        )


def add_defaultAnchors(writer, image, anchors, epoch=0):
    if isinstance(image, torch.Tensor):
        image = (image * 255).int().permute(1, 2, 0).cpu().numpy().astype(np.uint8)

    size = np.array(image.shape[1::-1])
    for stride, anchor in anchors.items():
        image_show = image.copy()

        anchor_wh = anchor[:, 2:] - anchor[:, :2] + 1
        anchor_ctr = anchor[:, :2] + 0.5 * anchor_wh
        size_anchor = (size // stride) * stride
        x, y = torch.meshgrid(
            [torch.arange(0, size_anchor[i], stride, device="cpu") for i in range(2)]
        )
        xyxy = torch.stack((x, y, x, y), 2).view(-1, 4)

        xy = (xyxy[:, :2] + anchor_ctr[0]).int()
        for _xy in xy:
            cv2.circle(image_show, tuple(_xy.tolist()), 2, (255, 0, 0), -1)
        shift_anchor = (anchor + xyxy[xyxy.shape[0] // 2]).int().tolist()
        for an in shift_anchor:
            cv2.rectangle(image_show, tuple(an[:2]), tuple(an[2:]), (0, 255, 0), 1)
        writer.add_image(
            "anchors/stride_{}".format(stride), image_show, epoch, dataformats="HWC"
        )


def add_matchedAnchorsWithBox(writer, image, anchor, stride, depth, epoch=0):
    if isinstance(image, torch.Tensor):
        image = (image * 255).int().permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    if isinstance(depth, torch.Tensor):
        depth = depth.clamp(-1, 1).cpu().numpy().astype(np.int8)

    size = np.array(image.shape[1::-1])
    anchor_wh = anchor[:, 2:] - anchor[:, :2] + 1
    anchor_ctr = anchor[:, :2] + 0.5 * anchor_wh
    size_anchor = (size // stride) * stride
    x, y = torch.meshgrid(
        [torch.arange(0, size_anchor[i], stride, device="cpu") for i in range(2)]
    )
    xyxy = torch.stack((x, y, x, y), 2).view(-1, 4)

    xy = (xyxy[:, :2] + anchor_ctr[0]).int()
    depth_xy = (xyxy[:, :2] // stride).int()
    color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # neg, pos, ignore

    for an_idx, an in enumerate(anchor):
        image_show = image.copy()
        for _xy, _xyxy, _depth_xy in zip(xy, xyxy, depth_xy):
            _depth = depth[an_idx, 0, _depth_xy[1], _depth_xy[0]]
            cv2.circle(image_show, tuple(_xy.tolist()), 2, color[_depth], -1)
        writer.add_image(
            "matched_anchors/stride_{}_anchor_{}".format(stride, an_idx),
            image_show,
            epoch,
            dataformats="HWC",
        )


def add_imagesWithBoxes(writer, tag, images, boxes, class_names=[], epoch=0):
    if isinstance(images, torch.Tensor):
        images = (images * 255).int().permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        boxes = boxes.int().cpu().numpy()
    for i, (image, box) in enumerate(zip(images, boxes)):
        image = np.ascontiguousarray(image)
        for b in box:
            if b[4] == -1:
                continue
            cv2.rectangle(image, tuple(b[:2]), tuple(b[2:4]), (0, 255, 0), 1)
            c = class_names[b[4]] if class_names else b[4]
            cv2.putText(
                image,
                str(c),
                tuple(b[:2]),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        writer.add_image(tag + "/{}".format(i), image, epoch, dataformats="HWC")


def add_imagesWithMatchedBoxes(
    writer, tag, images, boxes, targets, class_names=[], epoch=0
):
    if isinstance(images, torch.Tensor):
        images = (images * 255).int().permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        boxes = boxes.cpu().detach().numpy()
        targets = targets.int().cpu().numpy()
    for i, (image, box, target) in enumerate(zip(images, boxes, targets)):
        image = np.ascontiguousarray(image)
        box = box[box[:, 4] > 0.5]
        iou_c = matrix_iou(box[:, :4], target[:, :4])
        matched = np.any(iou_c > 0.6, axis=1)
        for b in box[matched].astype(int):
            cv2.rectangle(image, tuple(b[:2]), tuple(b[2:4]), (255, 0, 255), 1)
        for b in target:
            if b[4] == -1:
                continue
            cv2.rectangle(image, tuple(b[:2]), tuple(b[2:4]), (0, 255, 0), 1)
            c = class_names[b[4]] if class_names else b[4]
            cv2.putText(
                image,
                str(c),
                tuple(b[:2]),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        writer.add_image(tag + "/{}".format(i), image, epoch, dataformats="HWC")


def add_anchorStrategy(writer, targets, num_thresholds=100):
    scale = torch.sqrt(targets[:, 2] * targets[:, 3]).cpu().numpy()
    ratio = (targets[:, 3] / targets[:, 2]).cpu().numpy()  # h/w
    scale[scale > 1000] = -1
    ratio[np.isinf(ratio)] = -1

    scale.sort(), ratio.sort()

    import matplotlib.pyplot as plt

    plt.switch_backend("agg")
    plt.style.use("ggplot")

    fig = plt.figure()
    plt.hist(scale, bins=num_thresholds)
    plt.xlabel("scale")
    plt.ylabel("frequence")
    # plt.xticks((np.arange(num_thresholds+1)[::-1]/num_thresholds+1) * scale.max())
    writer.add_figure("archor_strategy/scale_distribute", fig)
    fig.clf()

    fig = plt.figure()
    plt.hist(ratio, bins=num_thresholds)
    plt.xlabel("ratio")
    plt.ylabel("frequence")
    # plt.xticks([0.2,0.25,0.333,0.5,1,2,3,4,5])
    writer.add_figure("archor_strategy/ratio_distribute", fig)
    fig.clf()


def add_matchedAnchor(writer):
    raise NotImplementedError
