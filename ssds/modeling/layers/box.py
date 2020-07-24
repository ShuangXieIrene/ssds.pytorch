import torch

# from ssds._C import decode as decode_cuda
# from ssds._C import nms as nms_cuda
INF = 100000


def configure_ratio_scale(num_featmaps, ratios, scales):
    r""" Get the apect ratio and scale for the default anchor boxes

    From v.15, ssds.pytorch does not consider the enlarged 1:1 anchor box in the original ssd or tf version
    """
    if len(scales) == num_featmaps:
        scales = scales
    # for the current version of generate anchor, this is not make sense
    # elif len(cfg.SIZES) == 2:
    #     num_layers = len(strides)
    #     min_scale, max_scale = cfg.SIZES
    #     scales = [min_scale + (max_scale - min_scale) * i / (num_layers - 1) for i in range(num_layers)]
    else:
        raise ValueError(
            "cfg.SIZES is not correct,"
            "the len of cfg.SIZES should equal to num layers({}) or 2, but it is {}".format(
                num_featmaps, len(scales)
            )
        )
    for i in range(num_featmaps):
        if not isinstance(scales[i], list):
            scales[i] = [scales[i]]

    if isinstance(ratios[0], list):
        if len(ratios) == num_featmaps:
            ratios = ratios
        else:
            raise ValueError(
                "When cfg.ASPECT_RATIOS contains list for each layer,"
                "Len of cfg.ASPECT_RATIOS should equal to num layers({}), but it is {}".format(
                    num_featmaps, len(ratios)
                )
            )
    else:
        ratios = [ratios for _ in range(num_featmaps)]
    return ratios, scales


def generate_anchors(stride, ratio_vals, scales_vals):
    "Generate anchors coordinates from scales/ratios"

    scales = torch.FloatTensor(scales_vals).repeat(len(ratio_vals), 1)
    scales = scales.transpose(0, 1).contiguous().view(-1, 1)
    ratios = torch.FloatTensor(ratio_vals * len(scales_vals))

    wh = torch.FloatTensor([stride]).repeat(len(ratios), 2)
    ws = torch.round(torch.sqrt(wh[:, 0] * wh[:, 1] / ratios))
    dwh = torch.stack([ws, torch.round(ws * ratios)], dim=1)
    xy1 = 0.5 * (wh - dwh * scales)
    xy2 = 0.5 * (wh + dwh * scales) - 1
    return torch.cat([xy1, xy2], dim=1)


def box2delta(boxes, anchors):
    "Convert boxes to deltas from anchors"

    anchors_wh = anchors[:, 2:] - anchors[:, :2] + 1
    anchors_ctr = anchors[:, :2] + 0.5 * anchors_wh
    boxes_wh = boxes[:, 2:] - boxes[:, :2] + 1
    boxes_ctr = boxes[:, :2] + 0.5 * boxes_wh

    return torch.cat(
        [(boxes_ctr - anchors_ctr) / anchors_wh, torch.log(boxes_wh / anchors_wh)], 1
    )


def delta2box(deltas, anchors, size, stride):
    "Convert deltas from anchors to boxes"

    anchors_wh = anchors[:, 2:] - anchors[:, :2] + 1
    ctr = anchors[:, :2] + 0.5 * anchors_wh
    pred_ctr = deltas[:, :2] * anchors_wh + ctr
    pred_wh = torch.exp(deltas[:, 2:]) * anchors_wh

    m = torch.zeros([2], device=deltas.device, dtype=deltas.dtype)
    M = torch.tensor([size], device=deltas.device, dtype=deltas.dtype) * stride - 1
    clamp = lambda t: torch.max(m, torch.min(t, M))
    return torch.cat(
        [clamp(pred_ctr - 0.5 * pred_wh), clamp(pred_ctr + 0.5 * pred_wh - 1)], 1
    )


def get_sample_region(boxes, stride, anchor_points, radius=1.5):
    """
    This code is from
    https://github.com/yqyao/FCOS_PLUS/blob/0d20ba34ccc316650d8c30febb2eb40cb6eaae37/
    maskrcnn_benchmark/modeling/rpn/fcos/loss.py#L42
    """
    # get mins and maxs value for center boarder
    stride = stride * radius
    center = (boxes[:, :2] + boxes[:, 2:]) / 2
    center_boxes = torch.cat((center - stride, center + stride), dim=-1)

    # generate the difference between grid points and center boarder
    # to check whether it is located in the center areas
    lt = (
        anchor_points[:, :, None, :]
        - torch.max(center_boxes[:, :2], boxes[:, :2])[None, None, :]
    )
    rb = (
        torch.min(center_boxes[:, 2:], boxes[:, 2:])[None, None, :]
        - anchor_points[:, :, None, :]
    )
    center_boxes = torch.cat((lt, rb), -1)
    inside_boxes_mask = center_boxes.min(-1)[0] > 0
    return inside_boxes_mask


def snap_to_anchors_by_iou(
    boxes,
    size,
    stride,
    anchors,
    num_classes,
    match,
    center_sampling_radius,
    is_centerness,
    device,
):
    "Snap target boxes (x, y, w, h) to anchors by the iou between target boxes and anchors"

    num_anchors = anchors.size()[0] if anchors is not None else 1
    width, height = (int(size[0] / stride), int(size[1] / stride))

    if boxes.nelement() == 0:
        if is_centerness:
            return (
                torch.zeros([num_anchors, num_classes, height, width], device=device),
                torch.zeros([num_anchors, 4, height, width], device=device),
                torch.zeros([num_anchors, 1, height, width], device=device),
                torch.zeros([num_anchors, 1, height, width], device=device),
            )
        else:
            return (
                torch.zeros([num_anchors, num_classes, height, width], device=device),
                torch.zeros([num_anchors, 4, height, width], device=device),
                torch.zeros([num_anchors, 1, height, width], device=device),
            )

    boxes, classes = boxes.split(4, dim=1)
    match_threshold, unmatch_threshold = match

    # Generate anchors
    x, y = torch.meshgrid(
        [
            torch.arange(0, size[i], stride, device=device, dtype=classes.dtype)
            for i in range(2)
        ]
    )
    xyxy = torch.stack((x, y, x, y), 2).unsqueeze(0)
    anchors = anchors.view(-1, 1, 1, 4).to(dtype=classes.dtype)
    anchors = (xyxy + anchors).contiguous().view(-1, 4)

    # Compute overlap between boxes and anchors
    boxes = torch.cat([boxes[:, :2], boxes[:, :2] + boxes[:, 2:] - 1], 1)
    xy1 = torch.max(anchors[:, None, :2], boxes[:, :2])
    xy2 = torch.min(anchors[:, None, 2:], boxes[:, 2:])
    inter = torch.prod((xy2 - xy1 + 1).clamp(0), 2)
    boxes_area = torch.prod(boxes[:, 2:] - boxes[:, :2] + 1, 1)
    anchors_area = torch.prod(anchors[:, 2:] - anchors[:, :2] + 1, 1)
    overlap = inter / (anchors_area[:, None] + boxes_area - inter)

    # Keep best box per anchor
    overlap, indices = overlap.max(1)
    box_target = box2delta(boxes[indices], anchors)
    box_target = box_target.view(num_anchors, 1, width, height, 4)
    box_target = box_target.transpose(1, 4).transpose(2, 3)
    box_target = box_target.squeeze().contiguous()

    depth = torch.ones_like(overlap) * -1
    depth[overlap < unmatch_threshold] = 0  # background
    depth[overlap >= match_threshold] = (
        classes[indices][overlap >= match_threshold].squeeze() + 1
    )  # objects
    depth = depth.view(num_anchors, width, height)
    # center_sampling in ATSS
    if center_sampling_radius > 0:
        anchor_points = torch.stack((x, y), dim=2) + stride // 2
        inside_boxes_mask = (
            get_sample_region(boxes, stride, anchor_points, center_sampling_radius)
            .float()
            .max(-1)[0]
        )
        depth = torch.min(depth, inside_boxes_mask[None, ...])
    depth = depth.transpose(1, 2).contiguous()

    # Generate target classes
    cls_target = torch.zeros(
        (anchors.size()[0], num_classes + 1), device=device, dtype=boxes.dtype
    )
    if classes.nelement() == 0:
        classes = torch.LongTensor([num_classes], device=device).expand_as(indices)
    else:
        classes = classes[indices].long()
    classes = classes.view(-1, 1)
    classes[overlap < unmatch_threshold] = num_classes  # background has no class
    cls_target.scatter_(1, classes, 1)
    cls_target = cls_target[:, :num_classes].view(-1, 1, width, height, num_classes)
    cls_target = cls_target.transpose(1, 4).transpose(2, 3)
    cls_target = cls_target.squeeze().contiguous()

    if is_centerness:
        lt = torch.abs(box_target[:, :2] - 0.5 * torch.exp(box_target[:, 2:]))
        rb = torch.abs(box_target[:, :2] - 0.5 * torch.exp(box_target[:, 2:]))
        centerness = torch.sqrt(
            torch.prod(torch.min(lt, rb) / torch.max(lt, rb), dim=1)
        )
        return (
            cls_target.view(num_anchors, num_classes, height, width),
            box_target.view(num_anchors, 4, height, width),
            centerness.view(num_anchors, 1, height, width),
            depth.view(num_anchors, 1, height, width),
        )
    else:
        return (
            cls_target.view(num_anchors, num_classes, height, width),
            box_target.view(num_anchors, 4, height, width),
            depth.view(num_anchors, 1, height, width),
        )


def snap_to_anchors_by_scale(
    boxes,
    size,
    stride,
    anchors,
    num_classes,
    match,
    center_sampling_radius,
    is_centerness,
    device,
):
    "Snap target boxes (x, y, w, h) to anchors by the scale of target boxes"

    num_anchors = anchors.size()[0] if anchors is not None else 1
    width, height = (int(size[0] / stride), int(size[1] / stride))

    if boxes.nelement() == 0:
        if is_centerness:
            return (
                torch.zeros([num_anchors, num_classes, height, width], device=device),
                torch.zeros([num_anchors, 4, height, width], device=device),
                torch.zeros([num_anchors, 1, height, width], device=device),
                torch.zeros([num_anchors, 1, height, width], device=device),
            )
        else:
            return (
                torch.zeros([num_anchors, num_classes, height, width], device=device),
                torch.zeros([num_anchors, 4, height, width], device=device),
                torch.zeros([num_anchors, 1, height, width], device=device),
            )

    boxes, classes = boxes.split(4, dim=1)

    # generate threshold for each anchor
    anchors_wh = anchors[:, 2:] - anchors[:, :2] + 1
    anchors_size = torch.sqrt(torch.prod(anchors_wh, dim=1)).unsqueeze(1).unsqueeze(2)
    lower_threshold = (match[0] * anchors_size).clamp(-1)
    upper_threshold = match[1] * anchors_size

    # Generate anchors
    x, y = torch.meshgrid(
        [
            torch.arange(0, size[i], stride, device=device, dtype=classes.dtype)
            for i in range(2)
        ]
    )
    xyxy = torch.stack((x, y, x, y), 2).unsqueeze(0)
    anchors = anchors.view(-1, 1, 1, 4).to(dtype=classes.dtype)
    anchors = (xyxy + anchors).contiguous().view(-1, 4)
    anchor_points = (torch.stack((x, y), dim=2) + stride // 2).to(dtype=classes.dtype)

    # Compute overlap between boxes and anchors
    boxes = torch.cat([boxes[:, :2], boxes[:, :2] + boxes[:, 2:] - 1], 1)
    boxes_area = torch.sqrt(torch.prod(boxes[:, 2:] - boxes[:, :2] + 1, 1))

    # limit the positive sample anchor points inside of the box or center box
    if center_sampling_radius > 0:
        # limit the box size range for each location
        is_cared_in_the_level = (boxes_area >= lower_threshold) & (
            boxes_area <= upper_threshold
        )
        # center_sampling in ATSS
        anchor_points = torch.stack((x, y), dim=2) + stride // 2
        inside_boxes_mask = get_sample_region(boxes, stride, anchor_points).view(
            -1, boxes.shape[0]
        )
    else:
        anchor_points = (torch.stack((x, y), dim=2) + stride // 2).view(-1, 2)
        lt = anchor_points[:, None, :] - boxes[:, :2]
        rb = boxes[:, 2:] - anchor_points[:, None, :]
        box_target = torch.cat([lt, rb], dim=-1)
        # limit the regression range for each location
        max_box_target = box_target.max(dim=-1)[0]
        is_cared_in_the_level = (max_box_target >= lower_threshold) & (
            max_box_target <= upper_threshold
        )
        # no center sampling, it will use all the points within a ground-truth box
        inside_boxes_mask = box_target.min(dim=-1)[0] > 0

    # if there are still more than one objects for a location,
    # we choose the one with minimal area
    mask = (is_cared_in_the_level & inside_boxes_mask).view(-1, boxes.shape[0])
    boxes_area = boxes_area.repeat(mask.shape[0], 1)
    boxes_area[mask == 0] = INF
    mask, _ = mask.max(dim=1)
    min_area, indices = boxes_area.min(dim=1)

    # Keep best box per anchor
    box_target = box2delta(boxes[indices], anchors)
    box_target = box_target.view(num_anchors, 1, width, height, 4)
    box_target = box_target.transpose(1, 4).transpose(2, 3)
    box_target = box_target.squeeze().contiguous()

    depth = torch.ones_like(mask, dtype=classes.dtype) * -1
    depth[mask == 0] = 0  # background
    depth[mask != 0] = classes[indices][mask != 0].squeeze() + 1  # objects
    depth = depth.view(num_anchors, width, height).transpose(1, 2).contiguous()

    # Generate target classes
    cls_target = torch.zeros(
        (anchors.size()[0], num_classes + 1), device=device, dtype=boxes.dtype
    )
    if classes.nelement() == 0:
        classes = torch.LongTensor([num_classes], device=device).expand_as(indices)
    else:
        classes = classes[indices].long()
    classes = classes.view(-1, 1)
    classes[mask == 0] = num_classes  # background has no class
    cls_target.scatter_(1, classes, 1)
    cls_target = cls_target[:, :num_classes].view(-1, 1, width, height, num_classes)
    cls_target = cls_target.transpose(1, 4).transpose(2, 3)
    cls_target = cls_target.squeeze().contiguous()

    if is_centerness:
        lt = torch.abs(box_target[:, :2] - 0.5 * torch.exp(deltas[:, 2:]))
        rb = torch.abs(box_target[:, :2] - 0.5 * torch.exp(deltas[:, 2:]))
        centerness = torch.sqrt(
            torch.prod(torch.min(lt, rb) / torch.max(lt, rb), dim=1)
        )
        return (
            cls_target.view(num_anchors, num_classes, height, width),
            box_target.view(num_anchors, 4, height, width),
            centerness.view(num_anchors, 1, height, width),
            depth.view(num_anchors, 1, height, width),
        )
    else:
        return (
            cls_target.view(num_anchors, num_classes, height, width),
            box_target.view(num_anchors, 4, height, width),
            depth.view(num_anchors, 1, height, width),
        )


def extract_targets(
    targets,
    anchors,
    classes,
    stride,
    size,
    match=[0.5, 0.4],
    center_sampling_radius=0,
    is_centerness=False,
):
    "snap the targets to anchors"
    cls_target, box_target, depth = [], [], []
    for target in targets:
        target = target[target[:, -1] > -1]
        if isinstance(match[0], float):
            snapped = snap_to_anchors_by_iou(
                target,
                [s * stride for s in size[::-1]],
                stride,
                anchors[stride].to(targets.device),
                classes,
                match,
                center_sampling_radius,
                is_centerness,
                targets.device,
            )
        elif isinstance(match[0], list):
            idx = list(anchors).index(stride)
            snapped = snap_to_anchors_by_scale(
                target,
                [s * stride for s in size[::-1]],
                stride,
                anchors[stride].to(targets.device),
                classes,
                match[idx],
                center_sampling_radius,
                is_centerness,
                targets.device,
            )
        else:
            raise ValueError("unvalidate match param")
        for l, s in zip((cls_target, box_target, depth), snapped):
            l.append(s)
    return torch.stack(cls_target), torch.stack(box_target), torch.stack(depth)


def decode(
    all_cls_head,
    all_box_head,
    stride=1,
    threshold=0.05,
    top_n=1000,
    anchors=None,
    rescore=True,
):
    "Box Decoding and Filtering"

    # if torch.cuda.is_available():
    #     return decode_cuda(all_cls_head.float(), all_box_head.float(),
    #         anchors.view(-1).tolist(), stride, threshold, top_n)

    device = all_cls_head.device
    anchors = anchors.to(device).type(all_cls_head.type())
    num_anchors = anchors.size()[0] if anchors is not None else 1
    num_classes = all_cls_head.size()[1] // num_anchors
    height, width = all_cls_head.size()[-2:]

    batch_size = all_cls_head.size()[0]
    out_scores = torch.zeros((batch_size, top_n), device=device)
    out_boxes = torch.zeros((batch_size, top_n, 4), device=device)
    out_classes = torch.zeros((batch_size, top_n), device=device)

    # Per item in batch
    for batch in range(batch_size):
        cls_head = all_cls_head[batch, :, :, :].contiguous().view(-1)
        box_head = all_box_head[batch, :, :, :].contiguous().view(-1, 4)

        # Keep scores over threshold
        keep = (cls_head >= threshold).nonzero().view(-1)
        if keep.nelement() == 0:
            continue

        # Gather top elements
        scores = torch.index_select(cls_head, 0, keep)
        scores, indices = torch.topk(scores, min(top_n, keep.size()[0]), dim=0)
        indices = torch.index_select(keep, 0, indices).view(-1)
        classes = (indices / width / height) % num_classes
        classes = classes.type(all_cls_head.type())

        # Infer kept bboxes
        x = indices % width
        y = (indices / width) % height
        a = indices / num_classes / height / width
        box_head = box_head.view(num_anchors, 4, height, width)
        boxes = box_head[a, :, y, x]

        if anchors is not None:
            grid = (
                torch.stack([x, y, x, y], 1).type(all_cls_head.type()) * stride
                + anchors[a, :]
            )
            boxes = delta2box(boxes, grid, [width, height], stride)
            if rescore:
                grid_center = (grid[:, :2] + grid[:, 2:]) / 2
                lt = torch.abs(grid_center - boxes[:, :2])
                rb = torch.abs(boxes[:, 2:] - grid_center)
                centerness = torch.sqrt(
                    torch.prod(torch.min(lt, rb) / torch.max(lt, rb), dim=1)
                )
                scores = scores * centerness

        out_scores[batch, : scores.size()[0]] = scores
        out_boxes[batch, : boxes.size()[0], :] = boxes
        out_classes[batch, : classes.size()[0]] = classes

    return out_scores, out_boxes, out_classes


def nms(all_scores, all_boxes, all_classes, nms=0.5, ndetections=100, using_diou=True):
    "Non Maximum Suppression"

    # if torch.cuda.is_available():
    #     return nms_cuda(
    #         all_scores.float(), all_boxes.float(), all_classes.float(), nms, ndetections)

    device = all_scores.device
    batch_size = all_scores.size()[0]
    out_scores = torch.zeros((batch_size, ndetections), device=device)
    out_boxes = torch.zeros((batch_size, ndetections, 4), device=device)
    out_classes = torch.zeros((batch_size, ndetections), device=device)

    # Per item in batch
    for batch in range(batch_size):
        # Discard null scores
        keep = (all_scores[batch, :].view(-1) > 0).nonzero()
        scores = all_scores[batch, keep].view(-1)
        boxes = all_boxes[batch, keep, :].view(-1, 4)
        classes = all_classes[batch, keep].view(-1)

        if scores.nelement() == 0:
            continue

        # Sort boxes
        scores, indices = torch.sort(scores, descending=True)
        boxes, classes = boxes[indices], classes[indices]
        areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1).view(
            -1
        )
        keep = torch.ones(scores.nelement(), device=device, dtype=torch.uint8).view(-1)

        for i in range(ndetections):
            if i >= keep.nonzero().nelement() or i >= scores.nelement():
                i -= 1
                break

            # Find overlapping boxes with lower score
            xy1 = torch.max(boxes[:, :2], boxes[i, :2])
            xy2 = torch.min(boxes[:, 2:], boxes[i, 2:])
            inter = torch.prod((xy2 - xy1 + 1).clamp(0), 1)
            iou = inter / (areas + areas[i] - inter + 1e-7)

            if using_diou:
                outer_lt = torch.min(boxes[:, :2], boxes[i, :2])
                outer_rb = torch.max(boxes[:, 2:], boxes[i, 2:])

                inter_diag = ((boxes[:, :2] - boxes[i, :2]) ** 2).sum(dim=1)
                outer_diag = ((outer_rb - outer_lt) ** 2).sum(dim=1) + 1e-7
                diou = (iou - inter_diag / outer_diag).clamp(-1.0, 1.0)
                iou = diou

            criterion = (scores > scores[i]) | (iou <= nms) | (classes != classes[i])
            criterion[i] = 1

            # Only keep relevant boxes
            scores = scores[criterion.nonzero()].view(-1)
            boxes = boxes[criterion.nonzero(), :].view(-1, 4)
            classes = classes[criterion.nonzero()].view(-1)
            areas = areas[criterion.nonzero()].view(-1)
            keep[(~criterion).nonzero()] = 0

        out_scores[batch, : i + 1] = scores[: i + 1]
        out_boxes[batch, : i + 1, :] = boxes[: i + 1, :]
        out_classes[batch, : i + 1] = classes[: i + 1]

    return out_scores, out_boxes, out_classes
