from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import numpy as np
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

__C = edict()

cfg = __C

#
# Training options
#
__C.TRAIN = edict()

# Initial learning rate
__C.TRAIN.LEARNING_RATE = 0.001

# Momentum
__C.TRAIN.MOMENTUM = 0.9

# Weight decay, for regularization
__C.TRAIN.WEIGHT_DECAY = 0.0001

# Factor for reducing the learning rate
__C.TRAIN.GAMMA = 0.98

# Step size for reducing the learning rate
__C.TRAIN.STEPSIZE = 1

# Iteration intervals for showing the loss during training, on command line interface
__C.TRAIN.DISPLAY = 100

# Whether to initialize the weights with truncated normal distribution 
__C.TRAIN.TRUNCATED = False

# The number of checkpoints kept, older ones are deleted to save space
__C.TRAIN.CHECKPOINTS_KEPT = 10
__C.TRAIN.CHECKPOINTS_ITERS = 5000
__C.TRAIN.CHECKPOINTS_EPOCHS = 5

# The number of max iters
__C.TRAIN.MAX_EPOCHS = 5

# The time interval for saving tensorflow summaries
__C.TRAIN.SUMMARY_INTERVAL = 180

# Images to use per minibatch
__C.TRAIN.IMS_PER_BATCH = 1

# Minibatch size
__C.TRAIN.BATCH_SIZE = 128

# Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
__C.TRAIN.FG_THRESH = 0.5

# Overlap threshold for a ROI to be considered background (class = 0 if
# overlap in [LO, HI))
__C.TRAIN.BG_THRESH_HI = 0.5
__C.TRAIN.BG_THRESH_LO = 0.1

__C.TRAIN.TRAINABLE_SCOPE = 'base,extras,Norm,L2Norm,loc,conf'
__C.TRAIN.RESUME_SCOPE = 'base,extras,Norm,L2Norm,loc,conf'


#
# Testing options
#
__C.TEST = edict()

# Scale to use during testing (can NOT list multiple scales)
# The scale is the pixel size of an image's shortest side
__C.TEST.SCALES = (600,)

# Overlap threshold used for non-maximum suppression (suppress boxes with
# IoU >= this threshold)
__C.TEST.NMS = 0.3

#
# Model options
#
__C.MODEL = edict()

# Name of the base net used to extract the features
__C.MODEL.BASE_FN = 'vgg16'

# Name of the model used to detect boundingbox
__C.MODEL.MODEL_FN = 'ssd'

# image size for ssd
__C.MODEL.IMAGE_SIZE = [300, 300]

__C.MODEL.PIXEL_MEANS = (103.94, 116.78, 123.68)

__C.MODEL.PRIOR_BOX = edict()

__C.MODEL.PRIOR_BOX.STEPS = []

__C.MODEL.PRIOR_BOX.SIZES = [0.2, 0.95]

__C.MODEL.PRIOR_BOX.ASPECT_RATIOS = [[2,3], [2, 3], [2, 3], [2, 3], [2], [2]]

__C.MODEL.PRIOR_BOX.CLIP = True

__C.MODEL.FEATURE_LAYER = [22, 34, 'S', 'S', '', '']
__C.MODEL.LAYER_DEPTH = [-1, -1, 512, 256, 256, 256]
__C.MODEL.MBOX = [6, 6, 6, 6, 4, 4]
__C.MODEL.NUM_FUSED = 3 # used for fssd

# post process
__C.MODEL.POST_PROCESS = edict()
__C.MODEL.POST_PROCESS.NUM_CLASSES = 21
__C.MODEL.POST_PROCESS.BACKGROUND_LABEL = 0
__C.MODEL.POST_PROCESS.SCORE_THRESHOLD = 0.01
__C.MODEL.POST_PROCESS.IOU_THRESHOLD = 0.6
__C.MODEL.POST_PROCESS.MAX_DETECTIONS = 100

#
# MISC
#

# Root directory of project
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))

# image expand probability
__C.PROB = 0.6

# Data directory
__C.DATASET_FN = 'voc'
__C.DATA_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'data'))
__C.TRAIN_SETS = [('2012', 'trainval')]

# Place outputs model under an experiments directory
__C.EXP_DIR = './experiments/models/'

# Place outputs tensorboard log under an experiments directory
__C.LOG_DIR = './experiments/logs/'
__C.RESUME_CHECKPOINT = ''

def _merge_a_into_b(a, b):
  """Merge config dictionary a into config dictionary b, clobbering the
  options in b whenever they are also specified in a.
  """
  if type(a) is not edict:
    return

  for k, v in a.items():
    # a must specify keys that are in b
    if k not in b:
      raise KeyError('{} is not a valid config key'.format(k))

    # the types must match, too
    old_type = type(b[k])
    if old_type is not type(v):
      if isinstance(b[k], np.ndarray):
        v = np.array(v, dtype=b[k].dtype)
      else:
        raise ValueError(('Type mismatch ({} vs. {}) '
                          'for config key: {}').format(type(b[k]),
                                                       type(v), k))
    # recursively merge dicts
    if type(v) is edict:
      try:
        _merge_a_into_b(a[k], b[k])
      except:
        print(('Error under config key: {}'.format(k)))
        raise
    else:
      b[k] = v

def cfg_from_file(filename):
  """Load a config file and merge it into the default options."""
  import yaml
  with open(filename, 'r') as f:
    yaml_cfg = edict(yaml.load(f))

  _merge_a_into_b(yaml_cfg, __C)