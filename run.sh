#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=1


python train.py --cfg=./experiments/cfgs/fssd_lite_mobilenetv1_train_coco.yml