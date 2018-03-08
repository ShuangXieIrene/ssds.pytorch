from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from ast import literal_eval
import os
import os.path as osp
import numpy as np

"""config system.
This file specifies default config options. You should not
change values in this file. Instead, you should write a config file (in yaml)
and use merge_cfg_from_file(yaml_file) to load it and override the default
options.
"""

class AttrDict(dict):

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        elif name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        if name in self.__dict__:
            self.__dict__[name] = value
        else:
            self[name] = value

__C = AttrDict()

cfg = __C

# ---------------------------------------------------------------------------- #
# Model options
# ---------------------------------------------------------------------------- #
__C.MODEL = AttrDict()

# Name of the base net used to extract the features
__C.MODEL.NETS = 'vgg16'

# Name of the model used to detect boundingbox
__C.MODEL.SSDS = 'ssd'

# Whether use half precision for the model. currently only inference support.
__C.MODEL.HALF_PRECISION = True

# image size for ssd
__C.MODEL.IMAGE_SIZE = [300, 300]

# number of the class for the model
__C.MODEL.NUM_CLASSES = 21

# FEATURE_LAYER to extract the proposed bounding box, 
# the first dimension is the feature layer/type, 
# while the second dimension is feature map channel. 
__C.MODEL.FEATURE_LAYER = [[22, 34, 'S', 'S', '', ''], [512, 1024, 512, 256, 256, 256]]

# STEPS for the proposed bounding box, if empty the STEPS = image_size / feature_map_size
__C.MODEL.STEPS = []

# STEPS for the proposed bounding box, a list from min value to max value
__C.MODEL.SIZES = [0.2, 0.95]

# ASPECT_RATIOS for the proposed bounding box, 1 is default contains
__C.MODEL.ASPECT_RATIOS = [[2,3], [2, 3], [2, 3], [2, 3], [2], [2]]

# 
__C.MODEL.CLIP = True

# FSSD setting, NUM_FUSED for fssd
__C.MODEL.NUM_FUSED = 3


# ---------------------------------------------------------------------------- #
# Train options
# ---------------------------------------------------------------------------- #
__C.TRAIN = AttrDict()
# The number of checkpoints kept, older ones are deleted to save space
__C.TRAIN.CHECKPOINTS_KEPT = 10
__C.TRAIN.CHECKPOINTS_EPOCHS = 5
# The number of max iters
__C.TRAIN.MAX_EPOCHS = 300
# Minibatch size
__C.TRAIN.BATCH_SIZE = 128
# trainable scope and resuming scope
__C.TRAIN.TRAINABLE_SCOPE = 'base,extras,norm,loc,conf'
__C.TRAIN.RESUME_SCOPE = ''

# ---------------------------------------------------------------------------- #
# optimizer options
# ---------------------------------------------------------------------------- #
__C.TRAIN.OPTIMIZER = AttrDict()
# type of the optimizer
__C.TRAIN.OPTIMIZER.OPTIMIZER = 'sgd'
# Initial learning rate
__C.TRAIN.OPTIMIZER.LEARNING_RATE = 0.001
# Momentum
__C.TRAIN.OPTIMIZER.MOMENTUM = 0.9
# Momentum_2
__C.TRAIN.OPTIMIZER.MOMENTUM_2 = 0.99
# epsilon
__C.TRAIN.OPTIMIZER.EPS = 1e-8
# Weight decay, for regularization
__C.TRAIN.OPTIMIZER.WEIGHT_DECAY = 0.0001

# ---------------------------------------------------------------------------- #
# lr_scheduler options
# ---------------------------------------------------------------------------- #
__C.TRAIN.LR_SCHEDULER = AttrDict()
# type of the LR_SCHEDULER
__C.TRAIN.LR_SCHEDULER.SCHEDULER = 'step'
# Step size for reducing the learning rate
__C.TRAIN.LR_SCHEDULER.STEPS = [1]
# Factor for reducing the learning rate
__C.TRAIN.LR_SCHEDULER.GAMMA = 0.98
# warm_up epochs
__C.TRAIN.LR_SCHEDULER.WARM_UP_EPOCHS = 0
# The number of max iters
__C.TRAIN.LR_SCHEDULER.MAX_EPOCHS = __C.TRAIN.MAX_EPOCHS - __C.TRAIN.LR_SCHEDULER.WARM_UP_EPOCHS

# ---------------------------------------------------------------------------- #
# Test options
# ---------------------------------------------------------------------------- #
__C.TEST = AttrDict()
__C.TEST.BATCH_SIZE = __C.TRAIN.BATCH_SIZE
__C.TEST.TEST_SCOPE = [0, 300]


# ---------------------------------------------------------------------------- #
# Matcher options
# ---------------------------------------------------------------------------- #
# matcher
__C.MATCHER = AttrDict()
__C.MATCHER.NUM_CLASSES = __C.MODEL.NUM_CLASSES
__C.MATCHER.BACKGROUND_LABEL = 0
__C.MATCHER.MATCHED_THRESHOLD = 0.5
__C.MATCHER.UNMATCHED_THRESHOLD = 0.5
__C.MATCHER.NEGPOS_RATIO = 3
__C.MATCHER.VARIANCE = [0.1, 0.2]


# ---------------------------------------------------------------------------- #
# Post process options
# ---------------------------------------------------------------------------- #
# post process
__C.POST_PROCESS = AttrDict()
__C.POST_PROCESS.NUM_CLASSES = __C.MODEL.NUM_CLASSES
__C.POST_PROCESS.BACKGROUND_LABEL = __C.MATCHER.BACKGROUND_LABEL
__C.POST_PROCESS.SCORE_THRESHOLD = 0.01
__C.POST_PROCESS.IOU_THRESHOLD = 0.6
__C.POST_PROCESS.MAX_DETECTIONS = 100
__C.POST_PROCESS.VARIANCE = __C.MATCHER.VARIANCE 


# ---------------------------------------------------------------------------- #
# Dataset options
# ---------------------------------------------------------------------------- #
# Root directory of project
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))

__C.DATASET = AttrDict()
# name of the dataset
__C.DATASET.DATASET = 'voc'
# path of the dataset
__C.DATASET.DATASET_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'data'))
# train set scope
__C.DATASET.TRAIN_SETS = [('2007', 'trainval'), ('2012', 'trainval')]
# test set scope
__C.DATASET.TEST_SETS = [('2007', 'test')]
# image expand probability during train
__C.DATASET.PROB = 0.6
# image size
__C.DATASET.IMAGE_SIZE = __C.MODEL.IMAGE_SIZE
# image mean
__C.DATASET.PIXEL_MEANS = (103.94, 116.78, 123.68)
# train batch size
__C.DATASET.TRAIN_BATCH_SIZE = __C.TRAIN.BATCH_SIZE
# test batch size
__C.DATASET.TEST_BATCH_SIZE = __C.TEST.BATCH_SIZE
# number of workers to extract datas
__C.DATASET.NUM_WORKERS = 8


# ---------------------------------------------------------------------------- #
# Export options
# ---------------------------------------------------------------------------- #
# Place outputs model under an experiments directory
__C.EXP_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'experiments/models/'))
__C.LOG_DIR = __C.EXP_DIR
__C.RESUME_CHECKPOINT = ''
__C.CHECKPOINTS_PREFIX = '{}_{}_{}'.format(__C.MODEL.SSDS, __C.MODEL.NETS, __C.DATASET.DATASET)
__C.PHASE = ['train', 'eval', 'test']

# def _merge_a_into_b(a, b):
#   """Merge config dictionary a into config dictionary b, clobbering the
#   options in b whenever they are also specified in a.
#   """
#   if type(a) is not AttrDict:
#     return

#   for k, v in a.items():
#     # a must specify keys that are in b
#     if k not in b:
#       raise KeyError('{} is not a valid config key'.format(k))

#     # the types must match, too
#     old_type = type(b[k])
#     if old_type is not type(v):
#       if isinstance(b[k], np.ndarray):
#         v = np.array(v, dtype=b[k].dtype)
#       else:
#         raise ValueError(('Type mismatch ({} vs. {}) '
#                           'for config key: {}').format(type(b[k]),
#                                                        type(v), k))
#     # recursively merge dicts
#     if type(v) is AttrDict:
#       try:
#         _merge_a_into_b(a[k], b[k])
#       except:
#         print(('Error under config key: {}'.format(k)))
#         raise
#     else:
#       b[k] = v

def _merge_a_into_b(a, b, stack=None):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    assert isinstance(a, AttrDict), 'Argument `a` must be an AttrDict'
    assert isinstance(b, AttrDict), 'Argument `b` must be an AttrDict'

    for k, v_ in a.items():
        full_key = '.'.join(stack) + '.' + k if stack is not None else k
        # a must specify keys that are in b
        if k not in b:
          raise KeyError('Non-existent config key: {}'.format(full_key))

        v = _decode_cfg_value(v_)
        v = _check_and_coerce_cfg_value_type(v, b[k], k, full_key)

        # Recursively merge dicts
        if isinstance(v, AttrDict):
            try:
                stack_push = [k] if stack is None else stack + [k]
                _merge_a_into_b(v, b[k], stack=stack_push)
            except BaseException:
                raise
        else:
            b[k] = v


def update_cfg():
    __C.TRAIN.LR_SCHEDULER.MAX_EPOCHS = __C.TRAIN.MAX_EPOCHS - __C.TRAIN.LR_SCHEDULER.WARM_UP_EPOCHS
    __C.DATASET.IMAGE_SIZE = __C.MODEL.IMAGE_SIZE
    __C.DATASET.TRAIN_BATCH_SIZE = __C.TRAIN.BATCH_SIZE
    __C.DATASET.TEST_BATCH_SIZE = __C.TEST.BATCH_SIZE
    __C.MATCHER.NUM_CLASSES = __C.MODEL.NUM_CLASSES
    __C.POST_PROCESS.NUM_CLASSES = __C.MODEL.NUM_CLASSES
    __C.POST_PROCESS.BACKGROUND_LABEL = __C.MATCHER.BACKGROUND_LABEL
    __C.POST_PROCESS.VARIANCE = __C.MATCHER.VARIANCE 
    __C.CHECKPOINTS_PREFIX = '{}_{}_{}'.format(__C.MODEL.SSDS, __C.MODEL.NETS, __C.DATASET.DATASET)


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = AttrDict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)
    update_cfg()

def _decode_cfg_value(v):
    """Decodes a raw config value (e.g., from a yaml config files or command
    line argument) into a Python object.
    """
    # Configs parsed from raw yaml will contain dictionary keys that need to be
    # converted to AttrDict objects
    if isinstance(v, dict):
        return AttrDict(v)
    # All remaining processing is only applied to strings
    if not isinstance(v, str):
        return v
    # Try to interpret `v` as a:
    #   string, number, tuple, list, dict, boolean, or None
    try:
        v = literal_eval(v)
    # The following two excepts allow v to pass through when it represents a
    # string.
    #
    # Longer explanation:
    # The type of v is always a string (before calling literal_eval), but
    # sometimes it *represents* a string and other times a data structure, like
    # a list. In the case that v represents a string, what we got back from the
    # yaml parser is 'foo' *without quotes* (so, not '"foo"'). literal_eval is
    # ok with '"foo"', but will raise a ValueError if given 'foo'. In other
    # cases, like paths (v = 'foo/bar' and not v = '"foo/bar"'), literal_eval
    # will raise a SyntaxError.
    except ValueError:
        pass
    except SyntaxError:
        pass
    return v

def _check_and_coerce_cfg_value_type(value_a, value_b, key, full_key):
    """Checks that `value_a`, which is intended to replace `value_b` is of the
    right type. The type is correct if it matches exactly or is one of a few
    cases in which the type can be easily coerced.
    """
    # The types must match (with some exceptions)
    type_b = type(value_b)
    type_a = type(value_a)
    if type_a is type_b:
        return value_a

    # Exceptions: numpy arrays, strings, tuple<->list
    if isinstance(value_b, np.ndarray):
        value_a = np.array(value_a, dtype=value_b.dtype)
    elif isinstance(value_b, str):
        value_a = str(value_a)
    elif isinstance(value_a, tuple) and isinstance(value_b, list):
        value_a = list(value_a)
    elif isinstance(value_a, list) and isinstance(value_b, tuple):
        value_a = tuple(value_a)
    else:
        raise ValueError(
            'Type mismatch ({} vs. {}) with values ({} vs. {}) for config '
            'key: {}'.format(type_b, type_a, value_b, value_a, full_key)
        )
    return value_a