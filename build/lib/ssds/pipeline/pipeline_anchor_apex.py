import sys
import torch
import time
from datetime import timedelta

from apex import amp

import ssds.core.tools as tools
import ssds.core.visualize_funcs as vsf
from ssds.modeling.layers.box import extract_targets

CURSOR_UP_ONE = "\x1b[1A"
ERASE_LINE = "\x1b[2K"


class ModelWithLossBasic(torch.nn.Module):
    """ Class use to help the gpu memory becomes more balance in ddp model
    """

    def __init__(
        self,
        model,
        cls_criterion,
        loc_criterion,
        num_classes,
        match,
        center_sampling_radius,
    ):
        super(ModelWithLossBasic, self).__init__()
        self.model = model
        self.cls_criterion = cls_criterion
        self.loc_criterion = loc_criterion
        self.num_classes = num_classes
        self.match = match
        self.center_radius = center_sampling_radius

    def forward(self, images, targets, anchors):
        loc, conf = self.model(images)

        cls_losses, loc_losses, fg_targets = [], [], []
        for j, (stride, anchor) in enumerate(anchors.items()):
            size = conf[j].shape[-2:]
            conf_target, loc_target, depth = extract_targets(
                targets,
                anchors,
                self.num_classes,
                stride,
                size,
                self.match,
                self.center_radius,
            )
            fg_targets.append((depth > 0).sum().float().clamp(min=1))

            c = conf[j].view_as(conf_target).float()
            cls_mask = (depth >= 0).expand_as(conf_target).float()
            cls_loss = self.cls_criterion(c, conf_target, depth)
            cls_loss = cls_mask * cls_loss
            cls_losses.append(cls_loss.sum())

            l = loc[j].view_as(loc_target).float()
            loc_loss = self.loc_criterion(l, loc_target)
            loc_mask = (depth > 0).expand_as(loc_loss).float()
            loc_loss = loc_mask * loc_loss
            loc_losses.append(loc_loss.sum())

        fg_targets = torch.stack(fg_targets).sum()
        cls_loss = torch.stack(cls_losses).sum() / fg_targets
        loc_loss = torch.stack(loc_losses).sum() / fg_targets
        return cls_loss, loc_loss, cls_losses, loc_losses


def train_anchor_based_epoch(
    model, data_loader, optimizer, anchors, writer, epoch, device, local_rank
):
    model.train()
    title = "Train: "

    if local_rank == 0:
        loss_writer = {
            "loc_loss": tools.AverageMeter(),
            "cls_loss": tools.AverageMeter(),
        }
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
        start_time = time.time()
        dataset_len = len(data_loader)

    for batch_idx, (images, targets) in enumerate(data_loader):
        if images.device != device:
            images, targets = images.to(device), targets.to(device)
        if targets.dtype != torch.float:
            targets = targets.float()

        cls_loss, loc_loss, cls_losses, loc_losses = model(images, targets, anchors)
        if torch.isnan(loc_loss) or torch.isnan(cls_loss):
            continue
        if local_rank == 0:
            for j, (cl, ll) in enumerate(zip(cls_losses, loc_losses)):
                loss_writer["cls_loss_{}".format(j)].update(cl.item())
                loss_writer["loc_loss_{}".format(j)].update(ll.item())
            loss_writer["cls_loss"].update(cls_loss.item())
            loss_writer["loc_loss"].update(loc_loss.item())
            log = {
                "cls_loss": cls_loss.item(),
                "loc_loss": loc_loss.item(),
                "lr": optimizer.param_groups[0]["lr"],
            }

        optimizer.zero_grad()
        total_loss = cls_loss + loc_loss
        if total_loss.item() == float("Inf") or torch.isnan(total_loss):
            continue
        with amp.scale_loss(total_loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()

        if local_rank == 0:
            elapsed_time = time.time() - start_time
            estimat_time = elapsed_time * (dataset_len) / (batch_idx + 1)
            # log per iter
            print(
                title + tools.format_dict_of_loss(log),
                "|",
                batch_idx + 1,
                "/",
                dataset_len,
                "| Time:",
                timedelta(seconds=int(elapsed_time)),
                "/",
                timedelta(seconds=int(estimat_time)),
                "\r",
                end="",
            )
            sys.stdout.flush()

    if local_rank == 0:
        log = {"lr": optimizer.param_groups[0]["lr"]}
        log.update({k: v.avg for k, v in loss_writer.items()})
        print(
            CURSOR_UP_ONE
            + ERASE_LINE
            + "===>Avg Train: "
            + tools.format_dict_of_loss(log),
            " | Time: ",
            timedelta(seconds=int(time.time() - start_time)),
        )

        # log for tensorboard
        for key, value in log.items():
            writer.add_scalar("Train/{}".format(key), value, epoch)
        targets[:, :, 2:4] = targets[:, :, :2] + targets[:, :, 2:4]
        vsf.add_imagesWithBoxes(
            writer, "Train Image", images[:5], targets[:5], epoch=epoch
        )

    return
