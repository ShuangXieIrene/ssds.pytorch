import sys
from tqdm import tqdm
import torch

import ssds.core.tools as tools
import ssds.core.visualize_funcs as vsf
from ssds.core.evaluation_metrics import MeanAveragePrecision
from ssds.modeling.layers.box import extract_targets

CURSOR_UP_ONE = "\x1b[1A"
ERASE_LINE = "\x1b[2K"


def train_anchor_based_epoch(
    model,
    data_loader,
    optimizer,
    cls_criterion,
    loc_criterion,
    anchors,
    num_classes,
    match,
    center_sampling_radius,
    writer,
    epoch,
    device,
):
    r""" the pipeline for training
    """
    model.train()

    title = "Train: "
    progress = tqdm(
        tools.IteratorTimer(data_loader),
        total=len(data_loader),
        smoothing=0.9,
        miniters=1,
        leave=True,
        desc=title,
    )

    loss_writer = {"loc_loss": tools.AverageMeter(), "cls_loss": tools.AverageMeter()}
    loss_writer.update(
        {
            "loc_loss_{}".format(j): tools.AverageMeter()
            for j, _ in enumerate(anchors.items())
        }
    )
    loss_writer.update(
        {
            "cls_loss_{}".format(j): tools.AverageMeter()
            for j, _ in enumerate(anchors.items())
        }
    )

    for batch_idx, (images, targets) in enumerate(progress):
        if images.device != device:
            images, targets = images.to(device), targets.to(device)
        if targets.dtype != torch.float:
            targets = targets.float()

        loc, conf = model(images)

        cls_losses, loc_losses, fg_targets = [], [], []
        for j, (stride, anchor) in enumerate(anchors.items()):
            size = conf[j].shape[-2:]
            conf_target, loc_target, depth = extract_targets(
                targets,
                anchors,
                num_classes,
                stride,
                size,
                match,
                center_sampling_radius,
            )
            fg_targets.append((depth > 0).sum().float().clamp(min=1))

            c = conf[j].view_as(conf_target).float()
            cls_mask = (depth >= 0).expand_as(conf_target).float()
            cls_loss = cls_criterion(c, conf_target, depth)
            cls_loss = cls_mask * cls_loss
            cls_losses.append(cls_loss.sum())

            l = loc[j].view_as(loc_target).float()
            loc_loss = loc_criterion(l, loc_target)
            loc_mask = (depth > 0).expand_as(loc_loss).float()
            loc_loss = loc_mask * loc_loss
            loc_losses.append(loc_loss.sum())

            if torch.isnan(loc_loss.sum()) or torch.isnan(cls_loss.sum()):
                continue
            loss_writer["cls_loss_{}".format(j)].update(cls_losses[-1].item())
            loss_writer["loc_loss_{}".format(j)].update(loc_losses[-1].item())

        fg_targets = torch.stack(fg_targets).sum()
        cls_loss = torch.stack(cls_losses).sum() / fg_targets
        loc_loss = torch.stack(loc_losses).sum() / fg_targets
        if torch.isnan(loc_loss) or torch.isnan(cls_loss):
            continue
        loss_writer["cls_loss"].update(cls_loss.item())
        loss_writer["loc_loss"].update(loc_loss.item())

        log = {
            "cls_loss": cls_loss.item(),
            "loc_loss": loc_loss.item(),
            "lr": optimizer.param_groups[0]["lr"],
        }

        optimizer.zero_grad()
        total_loss = cls_loss + loc_loss
        if total_loss.item() == float("Inf"):
            continue
        total_loss.backward()
        optimizer.step()

        # log per iter
        progress.set_description(title + tools.format_dict_of_loss(log))
        progress.update(1)

    progress.close()
    log = {"lr": optimizer.param_groups[0]["lr"]}
    log.update({k: v.avg for k, v in loss_writer.items()})
    print(
        CURSOR_UP_ONE + ERASE_LINE + "===>Avg Train: " + tools.format_dict_of_loss(log)
    )

    # log for tensorboard
    for key, value in log.items():
        writer.add_scalar("Train/{}".format(key), value, epoch)
    targets[:, :, 2:4] = targets[:, :, :2] + targets[:, :, 2:4]
    vsf.add_imagesWithBoxes(writer, "Train Image", images[:5], targets[:5], epoch=epoch)

    return


def eval_anchor_based_epoch(
    model,
    data_loader,
    decoder,
    cls_criterion,
    loc_criterion,
    anchors,
    num_classes,
    writer,
    epoch,
    device,
):
    r""" the pipeline for evaluation
    """
    model.eval()
    title = "Eval: "
    progress = tqdm(
        tools.IteratorTimer(data_loader),
        total=len(data_loader),
        smoothing=0.9,
        miniters=1,
        leave=True,
        desc=title,
    )

    metric = MeanAveragePrecision(
        num_classes, decoder.conf_threshold, decoder.nms_threshold
    )
    for batch_idx, (images, targets) in enumerate(progress):
        if images.device != device:
            images, targets = images.to(device), targets.to(device)
        if targets.dtype != torch.float:
            targets = targets.float()

        loc, conf = model(images)

        # removed loss since the conf is sigmod in the evaluation stage,
        # the conf loss is not meaningful anymore
        detections = decoder(loc, conf, anchors)
        targets[:, :, 2:4] = targets[:, :, :2] + targets[:, :, 2:4]  # from xywh to ltrb
        metric(detections, targets)

        # log per iter
        progress.update(1)

    progress.close()
    mAP, (prec, rec, ap) = metric.get_results()

    log = {"mAP": mAP}
    if len(ap) < 5:
        for i, a in enumerate(ap):
            log["AP@cls{}".format(i)] = a
    print(
        CURSOR_UP_ONE + ERASE_LINE + "===>Avg Eval: " + tools.format_dict_of_loss(log)
    )

    # log for tensorboard
    for key, value in log.items():
        writer.add_scalar("Eval/{}".format(key), value, epoch)
    vsf.add_prCurve(writer, prec, rec, epoch=epoch)
    boxes = torch.cat((detections[1], detections[0][..., None]), dim=2)
    vsf.add_imagesWithMatchedBoxes(
        writer, "Eval Image", images[:5], boxes[:5], targets[:5], epoch=epoch
    )
    return
