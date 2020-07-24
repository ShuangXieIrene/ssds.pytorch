## Config System

config system for ssds.pytorch

### MODEL

| MODEL parameters | discription |
|---|---|
| MODEL.NETS | type of the backbone used to extract the features |
| MODEL.SSDS | type of the ssds model used to detect boundingbox |
| MODEL.IMAGE_SIZE | image size for ssd |
| MODEL.NUM_CLASSES | number of the class for the model |
| MODEL.FEATURE_LAYER | FEATURE_LAYER to extract the proposed bounding box, the first dimension is the feature layer/type, while the second dimension is feature map channel. |
| MODEL.SIZES | SIZES for the proposed anchor box, 1 is default contains |
| MODEL.ASPECT_RATIOS | ASPECT_RATIOS for the proposed anchor box, 1 is default contains |

### TRAIN

| TRAIN parameters | discription |
|---|---|
| TRAIN.BATCH_SIZE | batch size for training |
| TRAIN.TRAINABLE_SCOPE | trainable scope |
| TRAIN.RESUME_SCOPE | resuming scope |
| TRAIN.MAX_EPOCHS | the number of max epoch |
| TRAIN.CHECKPOINTS_EPOCHS | the number of interval epoch for checkpoints saving |
| TRAIN.CHECKPOINTS_KEPT | The number of checkpoints kept, older ones are deleted to save space |

#### TRAIN.OPTIMIZER

| TRAIN.OPTIMIZER parameters | discription |
|---|---|
| TRAIN.OPTIMIZER.OPTIMIZER | type of the optimizer |
| TRAIN.OPTIMIZER.LEARNING_RATE | Initial learning rate |
| TRAIN.OPTIMIZER.DIFFERENTIAL_LEARNING_RATE | Initial differential learning rate for different layers |
| TRAIN.OPTIMIZER.MOMENTUM | Momentum |
| TRAIN.OPTIMIZER.MOMENTUM_2 | Momentum_2 |
| TRAIN.OPTIMIZER.EPS | epsilon |
| TRAIN.OPTIMIZER.WEIGHT_DECAY | Weight decay, for regularization |

#### TRAIN.LR_SCHEDULER

| TRAIN.LR_SCHEDULER parameters | discription |
|---|---|
| TRAIN.LR_SCHEDULER.SCHEDULER | type of the LR_SCHEDULER |
| TRAIN.LR_SCHEDULER.STEPS | Step size for reducing the learning rate |
| TRAIN.LR_SCHEDULER.GAMMA | Factor for reducing the learning rate |
| TRAIN.LR_SCHEDULER.LR_MIN | min learning rate |

### TEST

| TEST parameters | discription |
|---|---|
| TEST.BATCH_SIZE | batch size for test |
| TEST.TEST_SCOPE | the epoch scope for test |

### POST_PROCESS

POST_PROCESS controls the parameter for ssds.modeling.layers.decoder.Decoder. which is used to decode the loc and conf feature maps
to predicted boxes.

| POST_PROCESS parameters | discription |
|---|---|
| POST_PROCESS.SCORE_THRESHOLD | the score threshold to filter the predict boxes, put it as 0.01 for evaluation |
| POST_PROCESS.IOU_THRESHOLD | the iou threshold to filter the predict boxes |
| POST_PROCESS.MAX_DETECTIONS | the max detection boxes for the final predicted output of ssds model |
| POST_PROCESS.MAX_DETECTIONS_PER_LEVEL | the max detection boxes for the each level output of ssds detect heads |
| POST_PROCESS.USE_DIOU | whether using diou to replace the iou in the nms part |
| POST_PROCESS.RESCORE_CENTER | whether rescore the boxes based on its anchor center location |

### DATASET

| DATASET parameters | discription |
|---|---|
| DATASET.DATASET | type of the dataset |
| DATASET.DATASET_DIR | path to the dataset folder |
| DATASET.TRAIN_SETS | train set scope |
| DATASET.TEST_SETS | test set scope |
| DATASET.PICKLE | whether use pickle to saved images and annotation (only works for Non-DALI dataset) |
| DATASET.NUM_WORKERS | 8 (only works for Non-DALI dataset) |
| DATASET.DEVICE_ID | the list of devices used to distributaed the data loading (only works for apex parrellel training)) |
| DATASET.MULTISCALE | list of image size used for multiscale training |


### DATASET.PREPROC

| DATASET.PREPROC parameters | discription |
|---|---|
| DATASET.PREPROC.MEAN | float, the mean for normalization |
| DATASET.PREPROC.STD | float, the std for normalization |
| DATASET.PREPROC.CROP_SCALE | list, the lower and upper bounder size for ssd random crop |
| DATASET.PREPROC.CROP_ASPECT_RATIO | list, the lower and upper bounder aspect ratio for ssd random crop |
| DATASET.PREPROC.CROP_ATTEMPTS | int, the numbder attempts to do the ssd random crop |
| DATASET.PREPROC.HUE_DELTA | float, hue delta |
| DATASET.PREPROC.BRI_DELTA | float, brightness delta |
| DATASET.PREPROC.CONTRAST_RANGE | list, the lower and upper bounder for contrast |
| DATASET.PREPROC.SATURATION_RANGE | list, the lower and upper bounder for saturation |
| DATASET.PREPROC.MAX_EXPAND_RATIO | float, the max expand ratio for padding |

### Others

| Others parameters | discription |
|---|---|
| EXP_DIR | the export dir |
| LOG_DIR | the log dir |
| RESUME_CHECKPOINT | The checkpoint used to resume |
| PHASE | The phases |
| DEVICE_ID | the list of devices used to distributaed the model training |