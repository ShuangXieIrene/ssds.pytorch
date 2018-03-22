#!/bin/bash

for file in ./experiments/cfgs/*.yml
do
  echo $file
  python demo.py --cfg=$file --demo=./experiments/person.jpg -t=time
done