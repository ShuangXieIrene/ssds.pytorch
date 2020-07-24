# ssds.pytorch
Repository for Single Shot MultiBox Detector and its variants, implemented with pytorch, python3. This repo is easy to setup and has plenty of visualization methods. We hope this repo can help people have a better understanding for ssd-like model and help people train and deploy the ssds model easily.

Currently, it contains these features:
- **Multiple SSD Variants**: ssd, fpn, bifpn, yolo and etc.
- **Multiple Base Network**: resnet, regnet, mobilenet and etc.
- **Visualize** the features of the ssd-like models to help the user understand the model design and performance.
- **Fast Training and Inference**: Utilize Nvidia Apex and Dali to fast training and support the user convert the model to ONNX or TensorRT for deployment.

This repo is depended on the work of [ODTK](https://github.com/NVIDIA/retinanet-examples), [Detectron](https://github.com/facebookresearch/Detectron) and [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection). Thanks for their works.

**Notice** The pretrain model for the current version does not finished yet, please check the [previous version](https://github.com/ShuangXieIrene/ssds.pytorch/tree/v0.3.1) for enrich pretrain models.

### Table of Contents
- <a href='#installation'>Installation</a>
- <a href='#usage'>Usage</a>
- <a href='#performance'>Performance and Model Zoo</a>
- <a href='#visualization'>Visualization</a>
- [Documentation](https://foreveryounggithub.github.io/ssds.doc)

## Installation
### requirements
* python>=3.7
* CUDA>=10.0
* pytorch>=1.4
### basic installation:
```bash
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
git clone https://github.com/ShuangXieIrene/ssds.pytorch.git
cd ssds.pytorch
python setup.py clean -a install
```
#### extra python libs for parallel training
Currently, nvidia DALI and apex is not include in the requirements.txt and need to install manually.

* [DALI](https://docs.nvidia.com/deeplearning/sdk/dali-developer-guide/docs/installation.html)
```bash
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/cuda/10.0 nvidia-dali
```
* [apex](https://github.com/NVIDIA/apex#linux)
```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

## Docker
```bash
git clone https://github.com/ShuangXieIrene/ssds.pytorch.git
docker build -t ssds:local ./ssds.pytorch/
docker run --gpus all -it --rm -v /data:/data ssds:local
```

## Usage
### 0. Check the config file by Visualization
Defined the network in a [config file](experiments/cfgs/tests/test.yml) and tweak the config file based on the visualized anchor boxes
```bash
python -m ssds.utils.visualize -cfg experiments/cfgs/tests/test.yml
```

### 1. Training
```bash
# basic training
python -m ssds.utils.train -cfg experiments/cfgs/tests/test.yml
# parallel training
python -m torch.distributed.launch --nproc_per_node={num_gpus} -m ssds.utils.train_ddp -cfg experiments/cfgs/tests/test.yml
```

### 2. Evaluation
```bash
python -m ssds.utils.train -cfg experiments/cfgs/tests/test.yml -e
```

### 3. Export to ONNX or TRT model
```bash
python -m ssds.utils.export -cfg experiments/cfgs/tests/test.yml -c best_mAP.pth -h
```

## Performance


## Visualization