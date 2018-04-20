#!/bin/bash

# test gpu speed
for file in ./experiments/cfgs/*.yml
do
  echo $file
  python demo.py --cfg=$file --demo=./experiments/person.jpg -t=time
done

# test cpu speed
# export CUDA_VISIBLE_DEVICES=''
# for file in ./experiments/cfgs/*.yml
# do
#   echo $file
#   python demo.py --cfg=$file --demo=./experiments/person.jpg -t=time
# done