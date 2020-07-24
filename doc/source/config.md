## Config System

config system for ssds.pytorch

### MODEL

* MODEL.NETS: type of the backbone used to extract the features

* MODEL.SSDS: type of the ssds model used to detect boundingbox

* MODEL.IMAGE_SIZE: image size for ssd

* MODEL.NUM_CLASSES: number of the class for the model

* MODEL.FEATURE_LAYER: FEATURE_LAYER to extract the proposed bounding box, the first dimension is the feature layer/type, while the second dimension is feature map channel.

* MODEL.SIZES: SIZES for the proposed anchor box, 1 is default contains

* MODEL.ASPECT_RATIOS: ASPECT_RATIOS for the proposed anchor box, 1 is default contains

### TRAIN

* TRAIN.TRAINABLE_SCOPE: trainable scope

* TRAIN.RESUME_SCOPE: resuming scope

* TRAIN.BATCH_SIZE: batch size for training 

* TRAIN.MAX_EPOCHS: the number of max epoch

* TRAIN.CHECKPOINTS_EPOCHS: the number of interval epoch for checkpoints saving

* TRAIN.CHECKPOINTS_KEPT: The number of checkpoints kept, older ones are deleted to save space


#### TRAIN.OPTIMIZER

* TRAIN.OPTIMIZER.OPTIMIZER: type of the optimizer

* TRAIN.OPTIMIZER.LEARNING_RATE: Initial learning rate

* TRAIN.OPTIMIZER.DIFFERENTIAL_LEARNING_RATE: Initial differential learning rate for different layers

* TRAIN.OPTIMIZER.MOMENTUM: Momentum

* TRAIN.OPTIMIZER.MOMENTUM_2: Momentum_2

* TRAIN.OPTIMIZER.EPS: epsilon

* TRAIN.OPTIMIZER.WEIGHT_DECAY: Weight decay, for regularization


#### TRAIN.LR_SCHEDULER

* TRAIN.LR_SCHEDULER.SCHEDULER: type of the LR_SCHEDULER

* TRAIN.LR_SCHEDULER.STEPS: Step size for reducing the learning rate

* TRAIN.LR_SCHEDULER.GAMMA: Factor for reducing the learning rate

* TRAIN.LR_SCHEDULER.LR_MIN: min learning rate


### TEST

* TEST.BATCH_SIZE: batch size for test 

* TEST.TEST_SCOPE: the epoch scope for test

### POST_PROCESS


* POST_PROCESS.SCORE_THRESHOLD: 0.01

* POST_PROCESS.IOU_THRESHOLD: 0.6

* POST_PROCESS.MAX_DETECTIONS: 100

* POST_PROCESS.MAX_DETECTIONS_PER_LEVEL: 300

* POST_PROCESS.USE_DIOU: True

* POST_PROCESS.RESCORE_CENTER: True