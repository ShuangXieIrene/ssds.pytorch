#!/bin/bash
set -e

sleep 1h
sleep 20m

python train.py --cfg=./experiments/cfgs/rfb_lite_mobilenetv2_train.yml 2>&1 | tee log.txt