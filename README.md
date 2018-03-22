# ssds.pytorch
Repository for Single Shot MultiBox Detector and its variants, implemented with pytorch, python3.

Currently, it contains these features:
- **Multiple SSD Variants**: ssd, rfb, fssd, ssd-lite, rfb-lite, fssd-lite
- **Multiple Base Network**: VGG, Mobilenet V1/V2
- **Free Image Size**
- **Visualization** with [tensorboard-pytorch](https://github.com/lanpa/tensorboard-pytorch): training loss, eval loss/mAP, example archor boxs.

This repo is depended on the work of [ssd.pytorch](https://github.com/amdegroot/ssd.pytorch), [faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch), [RFBNet](https://github.com/ruinmessi/RFBNet), [Detectron](https://github.com/facebookresearch/Detectron) and [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection). Thanks for there works.

### Table of Contents
- <a href='#installation'>Installation</a>
- <a href='#usage'>Usage</a>
- <a href='#performance'>Performance and Model Zoo</a>
- <a href='#visualization'>Visualization</a>
- <a href='#todo'>Future Work</a>
- <a href='#reference'>Reference</a>

## Installation
1. install [pytorch](http://pytorch.org/)
2. install requirements by `pip install -r ./requirements.txt`

## Usage
To train, test and demo some specific model. Please run the relative file in folder with the model configure file, like:

`python train.py --cfg=./experiments/cfgs/rfb_lite_mobilenetv2_train_voc.yml`

Change the configure file based on the note in [config_parse.py](./lib/utils/config_parse.py)

## Performance

| VOC2007     | SSD                                                                         | RFB                                                                         | FSSD                                                                        |
|-------------|-----------------------------------------------------------------------------|-----------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| VGG16       | [76.0%](https://drive.google.com/open?id=1TS50uVN-9_WJdyO1ImRAW0HFK11RkVlK) | [80.5%](https://drive.google.com/open?id=1bR79OsJY2cidjcI9L1DbXx2zde5sM2nf) | [77.8%](https://drive.google.com/open?id=1HPotrN0oM0oUQu_o-i_VYRYFlT3PKDrr) |
| MobilenetV1 | [72.7%](https://drive.google.com/open?id=1NMxw-bhvHTGThyNl-MKJrsou4n7HyDCG) | [73.7%](https://drive.google.com/open?id=1DWleN7Rcf92QYVAoeSxUeK7COXD4cuPN) | [78.4%](https://drive.google.com/open?id=1BVF7OaFcffJkqXbYBj1pj1nvX7ku8a55) |
| MobilenetV2 | [73.2%](https://drive.google.com/open?id=1SBeSIFv5z9AUtwcrJgPI4Xpc8-BCN6ro) | [73.4%](https://drive.google.com/open?id=1KUh1uvCJS_qEgq1r3t0VEYVge8K8tEzR) | [76.7%](https://drive.google.com/open?id=1t7kxurvfbXNYbFR64EULFSabWQpT256n) |


| COCO2017    | SSD                                                                         | RFB                                                                         | FSSD                                                                        |
|-------------|-----------------------------------------------------------------------------|-----------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| VGG16       | [25.4%](https://drive.google.com/open?id=1Bkt_nZW4fe_UrLyPOVltq0C7cTpABlQI) | [25.5%](https://drive.google.com/open?id=1j5G0dDuvofGh5POUPk0s9ys8by0wEDFL) | [24.5%](https://drive.google.com/open?id=1mxmv2Al5bzdBvNgY3disrZlxd6CSK2yh) |
| MobilenetV1 | [18.8%](https://drive.google.com/open?id=1yBpd3aIDvlK2j7HxsNj8kJuASTCaN5Bo) | [19.1%](https://drive.google.com/open?id=1SexO9XZFpMK2JGmr0mTGqosud-tb2wNe) | [24.2%](https://drive.google.com/open?id=1jRDwuXIeST4F5UBY2Z379nvkxz_OtxVd) |
| MobilenetV2 | [18.5%](https://drive.google.com/open?id=1FXjPnJ3X7PdH7Ii6lOEFA8EkKif4Ppnn) | [18.5%](https://drive.google.com/open?id=1uRfoi6iJo8Vd5yYMhzFJ97_l3NLtQhf-) | [22.2%](https://drive.google.com/open?id=1lOOjp4ZG1tkggSIbilT5ajKUJ-a-GRMK) |

| Net InferTime* (300px/fp32) | SSD    | RFB    | FSSD   |
|-----------------------------|--------|--------|--------|
| VGG16                       | 1.78ms | 4.20ms | 1.98ms |
| MobilenetV1                 | 2.87ms | 3.84ms | 2.62ms |
| MobilenetV2                 | 4.18ms | 5.28ms | 4.02ms |
*-only calculate the all network inference time, without pre-processing & post-processing. 
In fact, the speed of vgg is super impress me. Maybe it is caused by MobilenetV1 and MobilenetV2 is using -lite structure, which uses the seperate conv in the base and extra layers.

## Visualization

- visualize the network graph (terminal) -tensorboard has bugs.
![graph](./doc/imgs/graph.jpg)

- visualize the loss during the training progress and meanAP during the eval progress (terminal & tensorboard)
![train process](./doc/imgs/train_process.jpg)

- visualize archor box for each feature extractor (tensorboard)
![archor box](./doc/imgs/archor_box.jpg)

- visualize the preprocess steps for training (tensorboard)
![preprocess](./doc/imgs/preprocess.jpg)

- visualize the pr curve for different classes and anchor matching strategy for different properties during evaluation (tensorboard)
![pr_curve](./doc/imgs/pr_curve.jpg)
(*guess the dataset in the figure, coco or voc?)

- visualize weight (coming soon & any suggestion?)

## TODO
- add DSSDs: DSSD FPN TDM
- test the multi-resolution traning
- add rotation for prerprocess
- test focal loss
- add resnet, xception, inception
- figure out the problem of visualize graph
- speed up preprocess part (any suggestion?)
- speed up postprocess part (any suggestion?) huge bug!!!
<!-- - add half precision based on [csarofeen/examples](https://github.com/csarofeen/examples/tree/dist_fp16) -->
- add network visualization based on [pytorch-cnn-visualizations](https://github.com/utkuozbulak/pytorch-cnn-visualizations)
- [object detection](https://github.com/handong1587/handong1587.github.io/blob/master/_posts/deep_learning/2015-10-09-object-detection.md)
<!-- - convert to tensorrt based on [this](https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/topics/topics/workflows/manually_construct_tensorrt_engine.html) -->

## Reference
