import torch
import torch.utils.data as data
import numpy as np
import os
from glob import glob

from ssds import dataset


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).
    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations
    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    num_detections = []
    for img, target in batch:
        # for tup in sample:
        imgs.append(img)
        targets.append(target)
        num_detections.append(target.shape[0])

    torch_targets = -1 * torch.ones(
        [len(targets), max(max(num_detections), 1), 5], dtype=torch.float, device="cpu"
    )
    for i, target in enumerate(targets):
        num_dets = target.shape[0]
        torch_targets[i, :num_dets] = torch.from_numpy(target).float()
    return torch.stack(imgs, 0), torch_targets


def load_data(cfg, phase):
    r""" create the dataloader based on the config file.

    * If the phase == "train",
        it returns the dataloader in cfg.DATASET.TRAIN_SETS and fetch the randomly;
    * If the phase == "test",
        it returns the dataloader in cfg.DATASET.TEST_SETS and fetch the squentially;

    Args:
        cfg: the configs defined by cfg.DATASET
        phase (str): "train" or "test"

    Returns:
        dataloader 
    """
    training = phase == "train"
    image_sets = cfg.TRAIN_SETS if training else cfg.TEST_SETS
    batch_size = cfg.TRAIN_BATCH_SIZE if training else cfg.TEST_BATCH_SIZE

    if "Dali" in cfg.DATASET:
        data_loader = getattr(dataset, cfg.DATASET)(
            cfg=cfg,
            dataset_dir=cfg.DATASET_DIR,
            image_sets=image_sets,
            batch_size=batch_size,
            training=training,
        )
    else:
        _dataset = getattr(dataset, cfg.DATASET)(
            cfg=cfg,
            dataset_dir=cfg.DATASET_DIR,
            image_sets=image_sets,
            training=training,
        )
        data_loader = data.DataLoader(
            _dataset,
            batch_size,
            num_workers=cfg.NUM_WORKERS,
            shuffle=training,
            collate_fn=detection_collate,
            pin_memory=True,
        )
    return data_loader
