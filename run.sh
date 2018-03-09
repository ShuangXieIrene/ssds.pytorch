#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=1

python train.py --cfg=./experiments/cfgs/ssd_vgg16_train_coco.yml 2>&1 | tee log.txt