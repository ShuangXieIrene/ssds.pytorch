import os
import sys
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from ssds.core import optimizer, checkpoint, criterion, config, data_parallel
from ssds.modeling import model_builder
from ssds.pipeline.pipeline_anchor_basic import (
    train_anchor_based_epoch,
    eval_anchor_based_epoch,
)
from ssds.dataset.dataset_factory import load_data


class Solver(object):
    """
    A wrapper class for the training process
    """

    def __init__(self, cfg, render=False):
        self.cfg = cfg
        self.render = render

        # Build model
        print("===> Building model")
        self.model = model_builder.create_model(cfg.MODEL)

        # Utilize GPUs for computation
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Print the model architecture and parameters
        if self.render:
            print("Model architectures:\n{}\n".format(self.model))
            # print('Parameters and size:')
            # for name, param in self.model.named_parameters():
            #     print('{}: {}'.format(name, list(param.size())))

        # print trainable scope
        print("Trainable scope: {}".format(cfg.TRAIN.TRAINABLE_SCOPE))
        trainable_param_ = optimizer.trainable_param(
            self.model, cfg.TRAIN.TRAINABLE_SCOPE
        )
        self.optimizer = optimizer.configure_optimizer(
            trainable_param_, cfg.TRAIN.OPTIMIZER
        )
        self.lr_scheduler = optimizer.configure_lr_scheduler(
            self.optimizer, cfg.TRAIN.LR_SCHEDULER
        )
        self.max_epochs = cfg.TRAIN.MAX_EPOCHS

        # metric
        self.cls_criterion = getattr(criterion, self.cfg.MATCHER.CLASSIFY_LOSS)(
            alpha=self.cfg.MATCHER.FOCAL_ALPHA,
            gamma=self.cfg.MATCHER.FOCAL_GAMMA,
            negpos_ratio=self.cfg.MATCHER.NEGPOS_RATIO,
        )
        self.loc_criterion = getattr(criterion, self.cfg.MATCHER.LOCATE_LOSS)()

        # Set the logger
        self.writer = SummaryWriter(log_dir=cfg.LOG_DIR)

    def train_model(self):
        previous = checkpoint.find_previous_checkpoint(self.cfg.EXP_DIR)
        if previous:
            start_epoch = previous[0][-1]
            checkpoint.resume_checkpoint(
                self.model, previous[1][-1], self.cfg.TRAIN.RESUME_SCOPE
            )
        else:
            start_epoch = 0
            if self.cfg.RESUME_CHECKPOINT:
                print(
                    "Loading initial model weights from {:s}".format(
                        self.cfg.RESUME_CHECKPOINT
                    )
                )
                checkpoint.resume_checkpoint(
                    self.model, self.cfg.RESUME_CHECKPOINT, self.cfg.TRAIN.RESUME_SCOPE
                )

        if torch.cuda.is_available():
            print("Utilize GPUs for computation")
            print("Number of GPU available", torch.cuda.device_count())
            if len(self.cfg.DEVICE_ID) > 1:
                gpu0_bsz = self.cfg.TRAIN.BATCH_SIZE // (
                    8 * (len(self.cfg.DEVICE_ID) - 1) + 1
                )
                self.model = data_parallel.BalancedDataParallel(
                    gpu0_bsz, self.model, device_ids=self.cfg.DEVICE_ID
                )
                # self.model = nn.DataParallel(self.model, device_ids=self.cfg.DEVICE_ID)
            self.model.to(self.device)
            cudnn.benchmark = True

        # Load data
        print("===> Loading data")
        train_loader = load_data(cfg.DATASET, "train")
        eval_loader = load_data(cfg.DATASET, "eval") if "eval" in cfg.PHASE else None

        warm_up = self.cfg.TRAIN.LR_SCHEDULER.WARM_UP_EPOCHS
        for epoch in iter(range(start_epoch + 1, self.max_epochs + 1)):
            sys.stdout.write(
                "\rEpoch {epoch:d}/{max_epochs:d}:\n".format(
                    epoch=epoch, max_epochs=self.max_epochs
                )
            )
            torch.cuda.empty_cache()
            # start phases for epoch
            anchors = model_builder.create_anchors(
                self.cfg.MODEL, self.model, self.cfg.MODEL.IMAGE_SIZE, self.render
            )
            train_anchor_based_epoch(
                self.model,
                train_loader,
                self.optimizer,
                self.cls_criterion,
                self.loc_criterion,
                anchors,
                self.cfg.MODEL.NUM_CLASSES,
                self.cfg.MATCHER.MATCH_THRESHOLD,
                self.cfg.MATCHER.CENTER_SAMPLING_RADIUS,
                self.writer,
                epoch,
                self.device,
            )
            # save checkpoint
            if epoch % self.cfg.TRAIN.CHECKPOINTS_EPOCHS == 0:
                checkpoint.save_checkpoints(
                    self.model, self.cfg.EXP_DIR, self.cfg.CHECKPOINTS_PREFIX, epoch
                )
            # multi scale training
            if len(self.cfg.DATASET.MULTISCALE) > 1:
                batch_size, target_size = self.cfg.DATASET.MULTISCALE[
                    epoch % len(self.cfg.DATASET.MULTISCALE)
                ]
                train_loader.reset_size(batch_size, target_size)
            if "eval" in self.cfg.PHASE:
                anchors = model_builder.create_anchors(
                    self.cfg.MODEL, self.model, self.cfg.MODEL.IMAGE_SIZE
                )
                decoder = model_builder.create_decoder(self.cfg.POST_PROCESS)
                eval_anchor_based_epoch(
                    self.model,
                    eval_loader,
                    decoder,
                    self.cls_criterion,
                    self.loc_criterion,
                    anchors,
                    self.cfg.MODEL.NUM_CLASSES,
                    self.writer,
                    epoch,
                    self.device,
                )

            self.lr_scheduler.step()

    def eval_model(self):
        eval_loader = load_data(cfg.DATASET, "eval")
        self.model.to(self.device)
        anchors = model_builder.create_anchors(
            self.cfg.MODEL, self.model, self.cfg.MODEL.IMAGE_SIZE
        )
        decoder = model_builder.create_decoder(self.cfg.POST_PROCESS)

        previous = checkpoint.find_previous_checkpoint(self.cfg.EXP_DIR)
        if previous:
            for epoch, resume_checkpoint in zip(previous[0], previous[1]):
                if self.cfg.TEST.TEST_SCOPE[0] <= epoch <= self.cfg.TEST.TEST_SCOPE[1]:
                    checkpoint.resume_checkpoint(
                        self.model, resume_checkpoint, self.cfg.TRAIN.RESUME_SCOPE
                    )
                    eval_anchor_based_epoch(
                        self.model,
                        eval_loader,
                        decoder,
                        self.cls_criterion,
                        self.loc_criterion,
                        anchors,
                        self.cfg.MODEL.NUM_CLASSES,
                        self.writer,
                        epoch,
                        self.device,
                    )
        else:
            if self.cfg.RESUME_CHECKPOINT:
                print(
                    "Loading initial model weights from {:s}".format(
                        self.cfg.RESUME_CHECKPOINT
                    )
                )
                checkpoint.resume_checkpoint(
                    self.model, self.cfg.RESUME_CHECKPOINT, self.cfg.TRAIN.RESUME_SCOPE
                )
                eval_anchor_based_epoch(
                    self.model,
                    eval_loader,
                    decoder,
                    self.cls_criterion,
                    self.loc_criterion,
                    anchors,
                    self.cfg.MODEL.NUM_CLASSES,
                    self.writer,
                    0,
                    self.device,
                )


if __name__ == "__main__":
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Train/Eval a ssds.pytorch network")
    parser.add_argument(
        "-cfg",
        "--config",
        dest="config_file",
        help="optional config file",
        default=None,
        type=str,
    )
    parser.add_argument("-e", "--eval", action="store_true")
    parser.add_argument("-r", "--render", action="store_true")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

    cfg = config.cfg_from_file(args.config_file)
    solver = Solver(cfg, args.render)
    if args.eval:
        solver.eval_model()
    else:
        solver.train_model()
