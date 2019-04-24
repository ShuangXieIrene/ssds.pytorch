#!/bin/bash

rm time_benchmark.csv
cat <<EOF >./time_benchmark.csv
confg_file,total_time,preprocess_time,net_forward_time,detect_time,output_time
EOF


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