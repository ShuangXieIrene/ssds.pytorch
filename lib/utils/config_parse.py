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
__C.TRAIN.GAMMA = 0.1

# Iteration intervals for showing the loss during training, on command line interface
__C.TRAIN.DISPLAY = 100

# Whether to initialize the weights with truncated normal distribution 
__C.TRAIN.TRUNCATED = False

# The number of snapshots kept, older ones are deleted to save space
__C.TRAIN.SNAPSHOT_KEPT = 3

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

__C.MODEL.PRIOR_BOX = edict()

__C.MODEL.PRIOR_BOX.FEATURE_MAPS = [38, 19, 10, 5, 3, 1]

__C.MODEL.PRIOR_BOX.MIN_DIM = 300

__C.MODEL.PRIOR_BOX.STEPS = [8, 16, 32, 64, 100, 300]

__C.MODEL.PRIOR_BOX.MIN_SIZES = [30, 60, 111, 162, 213, 264]

__C.MODEL.PRIOR_BOX.MAX_SIZES = [60, 111, 162, 213, 264, 315]

__C.MODEL.PRIOR_BOX.ASPECT_RATIOS = [[2,3], [2, 3], [2, 3], [2, 3], [2], [2]]

__C.MODEL.PRIOR_BOX.VARIANCE = [0.1, 0.2]

__C.MODEL.PRIOR_BOX.CLIP = True

#
# MISC
#
__C.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])

# Root directory of project
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))

# Data directory
__C.DATA_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'data'))

# Place outputs under an experiments directory
__C.EXP_DIR = 'default'

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