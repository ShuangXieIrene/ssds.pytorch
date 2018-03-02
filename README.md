# ssds.pytorch
Repository for Single Shot MultiBox Detector and its variants, implemented with pytorch, python3.

Currently, it contains these features:
- **Multiple SSD Variants**: SSD, RFBNet, RFBNet-lite
- **Multiple Base Network**: VGG, Mobilenet V1/V2
- **Free Image Size**
- **Visualization** with [tensorboard-pytorch](https://github.com/lanpa/tensorboard-pytorch): training loss, eval loss/mAP, example archor boxs.

This repo is depended on the work of [ssd.pytorch](https://github.com/amdegroot/ssd.pytorch), [faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch), [RFBNet](https://github.com/ruinmessi/RFBNet), [Detectron](https://github.com/facebookresearch/Detectron) and [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection). Thanks for there works.

### Table of Contents
- <a href='#installation'>Installation</a>
- <a href='#usage'>Usage</a>
- <a href='#features'>Visual Features</a>
- <a href='#performance'>Performance and Model Zoo</a>
- <a href='#todo'>Future Work</a>
- <a href='#reference'>Reference</a>

## Installation

## Usage

## Performance

## TODO
- test the multi-resolution traning
- try rotation
- test ssd-lite, fssd
- test focal loss
- add resnet, xception, inception
- figure out the problem of visualize image in training, visualize graph, and visualize pr curve
- add half precision based on [csarofeen/examples](https://github.com/csarofeen/examples/tree/dist_fp16)
- add network visualization based on [pytorch-cnn-visualizations](https://github.com/utkuozbulak/pytorch-cnn-visualizations)

## Reference
