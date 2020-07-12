import torch
import os
import math
import ctypes
from contextlib import redirect_stdout
from pycocotools.coco import COCO

import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline

from .dali_dataiterator import DaliDataset, DaliPipeline

class COCOPipeline(Pipeline, DaliPipeline):
    'Dali pipeline for COCO'

    def __init__(self, image_dir, annotations_file, cache_path, batch_size, target_size, preproc_param, num_threads, num_shards, device_ids, training=False):
        Pipeline.__init__(self, batch_size=batch_size, num_threads=num_threads, device_id = device_ids, prefetch_queue_depth=num_threads, seed=42)
        DaliPipeline.__init__(self, target_size=target_size, preproc_param=preproc_param, training=training)

        self.reader = ops.COCOReader(annotations_file=annotations_file, file_root=image_dir, num_shards=num_shards, shard_id=0, 
                                     ltrb=True, ratio=True, shuffle_after_epoch=training, save_img_ids=True,
                                     dump_meta_files=True, dump_meta_files_path=cache_path)

    def define_graph(self):
        images, bboxes, labels, img_ids = self.reader()
        return self.predefined_graph(images, bboxes, labels)

class DaliCOCO(DaliDataset):
    'Data loader for data parallel using Dali for TFRecord files'
    def __init__(self, cfg, dataset_dir, image_sets, batch_size, training=False):
        super(DaliCOCO, self).__init__(cfg, dataset_dir, image_sets, batch_size, training)

        if len(image_sets) != 1:
            raise ValueError("For DaliCOCO dataset, the number of image_set has to be 1, currently it is {}".format(image_sets))
        
        self.image_dir = os.path.join(dataset_dir, "images", image_sets[0])
        self.annotations_file = os.path.join(dataset_dir, "annotations", "instances_{}.json".format(image_sets[0]))
        self.cache_path = os.path.join(dataset_dir, "cache")
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)

        self.pipe = COCOPipeline(image_dir=self.image_dir, annotations_file=self.annotations_file, cache_path=self.cache_path, **self.pipeline_args)
        self.pipe.build()

        with redirect_stdout(None):
            self.coco = COCO(self.annotations_file)
        self.ids = list(self.coco.imgs.keys())

    def __len__(self):
        return math.ceil(len(self.ids) // self.num_shards / self.batch_size)

    def reset_size(self, batch_size, target_size):
        self.batch_size = batch_size
        self.target_size = target_size
        self.pipeline_args["batch_size"] = batch_size
        self.pipeline_args["target_size"] = target_size

        del self.pipe

        self.pipe = COCOPipeline(image_dir=self.image_dir, annotations_file=self.annotations_file, cache_path=self.cache_path, **self.pipeline_args)
        self.pipe.build()