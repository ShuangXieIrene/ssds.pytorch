import os
import sys
import argparse
from tqdm import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter

from ssds.core import checkpoint, config
from ssds.modeling import model_builder
from ssds.dataset.dataset_factory import load_data
import ssds.core.visualize_funcs as vsf
import ssds.core.tools as tools

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="visualize a ssds.pytorch network")
    parser.add_argument(
        "-cfg",
        "--config",
        dest="config_file",
        help="optional config file",
        default=None,
        type=str,
    )
    parser.add_argument(
        "-a",
        "--anchor-strategy",
        help="analysis the anchor strategy in validate dataset",
        action="store_true",
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

    cfg = config.cfg_from_file(args.config_file)

    # Build model
    print("===> Building model")
    model = model_builder.create_model(cfg.MODEL)
    print("Model architectures:\n{}\n".format(model))
    anchors = model_builder.create_anchors(cfg.MODEL, model, cfg.MODEL.IMAGE_SIZE, True)
    decoder = model_builder.create_decoder(cfg.POST_PROCESS)

    print("Log details to {}".format(cfg.LOG_DIR))
    writer = SummaryWriter(log_dir=cfg.LOG_DIR)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if cfg.RESUME_CHECKPOINT:
        print("Loading initial model weights from {:s}".format(cfg.RESUME_CHECKPOINT))
        checkpoint.resume_checkpoint(model, cfg.RESUME_CHECKPOINT, "")

    model.eval().to(device)
    data_loader = load_data(cfg.DATASET, "train")

    images, targets = next(iter(data_loader))
    if images.device != device:
        images = images.to(device)
    loc, conf = model(images)
    detections = decoder(loc, conf, anchors)

    # visualize anchor
    if len(cfg.DATASET.MULTISCALE) > 1:
        # multi scale training
        for i in range(len(cfg.DATASET.MULTISCALE)):
            batch_size, target_size = cfg.DATASET.MULTISCALE[i]
            data_loader.reset_size(batch_size, target_size)
            images, targets = next(iter(data_loader))
            vsf.add_defaultAnchors(writer, images[0], anchors, epoch=i)
    else:
        vsf.add_defaultAnchors(writer, images[0], anchors, epoch=0)

    for j, (stride, anchor) in enumerate(anchors.items()):
        size = conf[j].shape[-2:]
        from ssds.modeling.layers.box import extract_targets

        _, _, depth = extract_targets(
            targets,
            anchors,
            cfg.MODEL.NUM_CLASSES,
            stride,
            size,
            cfg.MATCHER.MATCH_THRESHOLD,
            cfg.MATCHER.CENTER_SAMPLING_RADIUS,
        )
        for i in range(images.shape[0]):
            vsf.add_matchedAnchorsWithBox(
                writer, images[i], anchor, stride, depth[i], epoch=i
            )

    # visualize box
    targets[:, :, 2:4] = targets[:, :, :2] + targets[:, :, 2:4]
    boxes = torch.cat((detections[1], detections[0][..., None]), dim=2)
    vsf.add_imagesWithMatchedBoxes(
        writer, "Images", images[:5], boxes[:5], targets[:5], epoch=0
    )

    if args.anchor_strategy:
        data_loader = load_data(cfg.DATASET, "eval")
        title = "Load Data"
        progress = tqdm(
            tools.IteratorTimer(data_loader),
            total=len(data_loader),
            smoothing=0.9,
            miniters=1,
            leave=True,
            desc=title,
        )
        all_targets = []
        for images, targets in progress:
            targets = targets.view(-1, 5)
            targets = targets[targets[:, 4] != -1]
            all_targets.append(targets)
        all_targets = torch.cat(all_targets, dim=0)
        vsf.add_anchorStrategy(writer, all_targets)

    # visualize graph
    writer.add_graph(model, images)
    writer.close()
