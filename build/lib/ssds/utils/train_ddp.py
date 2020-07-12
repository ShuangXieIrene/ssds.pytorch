import os
import sys
import argparse
from pynvml.smi import nvidia_smi

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from apex import amp
from apex.parallel import convert_syncbn_model
from apex.parallel import DistributedDataParallel as DDP

from ssds.core import optimizer, checkpoint, criterion, config
from ssds.modeling import model_builder
from ssds.pipeline.pipeline_anchor_apex import (
    train_anchor_based_epoch,
    ModelWithLossBasic,
)
from ssds.dataset.dataset_factory import load_data

nvsmi = nvidia_smi.getInstance()


def getMemoryUsage(idx=0):
    usage = nvsmi.DeviceQuery("memory.used")["gpu"][idx]["fb_memory_usage"]
    return "%d %s" % (usage["used"], usage["unit"])


class Solver(object):
    """
    A wrapper class for the training process
    """

    def __init__(self, cfg, local_rank, render=False):
        self.cfg = cfg
        self.local_rank = local_rank
        self.render = render

        # Build model
        if self.local_rank == 0:
            print("===> Building model")
        self.model = model_builder.create_model(cfg.MODEL)
        self.load_model()

        # Utilize GPUs for computation
        self.device = torch.device("cuda:{}".format(local_rank))

        # Convert to sync model
        self.model = convert_syncbn_model(self.model)
        self.model.to(self.device)

        # Print the model architecture and parameters
        if self.render and self.local_rank == 0:
            print("Model architectures:\n{}\n".format(self.model))

        # print trainable scope
        if self.local_rank == 0:
            print("Trainable scope: {}".format(cfg.TRAIN.TRAINABLE_SCOPE))
        trainable_param_ = optimizer.trainable_param(
            self.model, cfg.TRAIN.TRAINABLE_SCOPE
        )
        self.optimizer = optimizer.configure_optimizer(
            trainable_param_, cfg.TRAIN.OPTIMIZER
        )

        # to apex version
        self.model, self.optimizer = amp.initialize(
            self.model, self.optimizer, opt_level="O1", loss_scale=128.0
        )  # keep_batchnorm_fp32 = True,

        # add scheduler
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
        self.writer = (
            SummaryWriter(log_dir=cfg.LOG_DIR) if self.local_rank == 0 else None
        )

    def train_model(self):
        modelWithLoss = ModelWithLossBasic(
            self.model,
            self.cls_criterion,
            self.loc_criterion,
            self.cfg.MODEL.NUM_CLASSES,
            self.cfg.MATCHER.MATCH_THRESHOLD,
            self.cfg.MATCHER.CENTER_SAMPLING_RADIUS,
        )

        if torch.cuda.is_available():
            print("Utilize GPUs for computation")
            print("Number of GPU available", torch.cuda.device_count())
            if self.cfg.DEVICE_ID:
                modelWithLoss = DDP(
                    modelWithLoss, delay_allreduce=True
                )  # , device_ids=self.cfg.DEVICE_ID)
            cudnn.benchmark = True

        # Load data
        if self.local_rank == 0:
            print("===> Loading data")
        train_loader = load_data(cfg.DATASET, "train")

        # multi scale training
        if len(self.cfg.DATASET.MULTISCALE) > 1:
            batch_size, target_size = self.cfg.DATASET.MULTISCALE[
                self.start_epoch % len(self.cfg.DATASET.MULTISCALE)
            ]
            train_loader.reset_size(batch_size, target_size)

        for epoch in iter(range(self.start_epoch + 1, self.max_epochs + 1)):
            if self.local_rank == 0:
                sys.stdout.write(
                    "\rEpoch {epoch:d}/{max_epochs:d}:\n".format(
                        epoch=epoch, max_epochs=self.max_epochs
                    )
                )
            torch.cuda.empty_cache()
            # start phases for epoch
            anchors = model_builder.create_anchors(
                self.cfg.MODEL,
                modelWithLoss.module.model,
                self.cfg.MODEL.IMAGE_SIZE,
                self.render,
            )
            train_anchor_based_epoch(
                modelWithLoss,
                train_loader,
                self.optimizer,
                anchors,
                self.writer,
                epoch,
                self.device,
                self.local_rank,
            )
            # save checkpoint
            if epoch % self.cfg.TRAIN.CHECKPOINTS_EPOCHS == 0 and self.local_rank == 0:
                checkpoint.save_checkpoints(
                    modelWithLoss.module.model,
                    self.cfg.EXP_DIR,
                    self.cfg.CHECKPOINTS_PREFIX,
                    epoch,
                )

            # multi scale training
            if len(self.cfg.DATASET.MULTISCALE) > 1:
                batch_size, target_size = self.cfg.DATASET.MULTISCALE[
                    epoch % len(self.cfg.DATASET.MULTISCALE)
                ]
                train_loader.reset_size(batch_size, target_size)

            if "eval" in self.cfg.PHASE:
                pass

            self.lr_scheduler.step()

    def load_model(self):
        previous = checkpoint.find_previous_checkpoint(self.cfg.EXP_DIR)
        if previous:
            self.start_epoch = previous[0][-1]
            self.model = checkpoint.resume_checkpoint(
                self.model, previous[1][-1], self.cfg.TRAIN.RESUME_SCOPE
            )
        else:
            self.start_epoch = 0
            if self.cfg.RESUME_CHECKPOINT:
                if self.local_rank == 0:
                    print(
                        "Loading initial model weights from {:s}".format(
                            self.cfg.RESUME_CHECKPOINT
                        )
                    )
                self.model = checkpoint.resume_checkpoint(
                    self.model, self.cfg.RESUME_CHECKPOINT, self.cfg.TRAIN.RESUME_SCOPE
                )

    def eval_model(self):
        return


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
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("-e", "--eval", action="store_true")
    parser.add_argument("-r", "--render", action="store_true")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend="nccl")  # , init_method='env://')

    cfg = config.cfg_from_file(args.config_file)
    solver = Solver(cfg, args.local_rank, args.render)
    solver.train_model()
