import torch
import os
import math
import ctypes
from subprocess import call
from glob import glob

import nvidia.dali.ops as ops
import nvidia.dali.types as types
import nvidia.dali.tfrecord as tfrec
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.plugin.pytorch as dalitorch

from .dali_dataiterator import DaliDataset, DaliPipeline

class TFRecordPipeline(Pipeline, DaliPipeline):
    """ Currently the easiest way for using dali to process the dataset is using TFRecord Files
    """
    def __init__(self, tfrecords, batch_size, target_size, preproc_param, num_threads, num_shards, device_ids, training=False):
        Pipeline.__init__(self, batch_size=batch_size, num_threads=num_threads, 
                          device_id=device_ids, prefetch_queue_depth=num_threads, seed=42,
                          exec_async=False, exec_pipelined=False)
        DaliPipeline.__init__(self, target_size=target_size, preproc_param=preproc_param, training=training)

        tfrecords_idx = [tfrecord+"_idx" for tfrecord in tfrecords]
        for tfrecord, tfrecord_idx in zip(tfrecords, tfrecords_idx):
            if os.path.exists(tfrecord_idx):
                continue
            call(["tfrecord2idx", tfrecord, tfrecord+"_idx"])
        self.length = sum([len(open(f).readlines()) for f in tfrecords_idx])

        self.input = ops.TFRecordReader(path = tfrecords,
                                        index_path = tfrecords_idx,
                                        features = {
                                         'image/height'  : tfrec.FixedLenFeature([1], tfrec.int64,  -1),
                                         'image/width'   : tfrec.FixedLenFeature([1], tfrec.int64,  -1),
                                         'image/encoded' : tfrec.FixedLenFeature((), tfrec.string, ""),
                                         'image/format'  : tfrec.FixedLenFeature((), tfrec.string, ""),
                                         'image/object/bbox/xmin':    tfrec.VarLenFeature(tfrec.float32, 0.0),
                                         'image/object/bbox/ymin':    tfrec.VarLenFeature(tfrec.float32, 0.0),
                                         'image/object/bbox/xmax':    tfrec.VarLenFeature(tfrec.float32, 0.0),
                                         'image/object/bbox/ymax':    tfrec.VarLenFeature(tfrec.float32, 0.0),
                                         'image/object/class/text':   tfrec.FixedLenFeature([ ], tfrec.string, ''),
                                         'image/object/class/label':  tfrec.VarLenFeature(tfrec.int64, -1)
                                         },
                                         num_shards = num_shards,
                                         random_shuffle = training)
        self.training = training
        self.cat = dalitorch.TorchPythonFunction(function=lambda l,t,r,b: torch.cat([l,t,r,b]).view(4,-1).permute(1,0)) #[l*w,t*h,r*w,b*h], [l,t,r,b]
        self.cast = ops.Cast(dtype=types.DALIDataType.INT32)

    def define_graph(self):
        inputs = self.input()
        images = inputs["image/encoded"]
        bboxes = self.cat(inputs["image/object/bbox/xmin"], inputs["image/object/bbox/ymin"],
                          inputs["image/object/bbox/xmax"], inputs["image/object/bbox/ymax"])
        labels = self.cast(inputs["image/object/class/label"])
        return self.predefined_graph(images, bboxes, labels)

    def __len__(self):
        return self.length


class DaliTFRecord(DaliDataset):
    'Data loader for data parallel using Dali for TFRecord files'
    def __init__(self, cfg, dataset_dir, image_sets, batch_size, training=False):
        super(DaliTFRecord, self).__init__(cfg, dataset_dir, image_sets, batch_size, training)

        self.tfrecords = [path for sets in image_sets for path in glob(os.path.join(dataset_dir, sets))]
        self.pipe = TFRecordPipeline(tfrecords=self.tfrecords, **self.pipeline_args)
        self.pipe.build()


    def reset_size(self, batch_size, target_size):
        self.batch_size = batch_size
        self.target_size = target_size
        self.pipeline_args["batch_size"] = batch_size
        self.pipeline_args["target_size"] = target_size

        del self.pipe

        self.pipe = TFRecordPipeline(tfrecords=self.tfrecords, **self.pipeline_args)
        self.pipe.build()