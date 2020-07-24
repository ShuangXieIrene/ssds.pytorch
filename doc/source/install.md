## Installation

### Python Installation

#### requirements
* python>=3.7
* CUDA>=10.0
* pytorch>=1.4
#### basic installation:
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

### Docker
```bash
git clone https://github.com/ShuangXieIrene/ssds.pytorch.git
docker build -t ssds:local ./ssds.pytorch/
docker run --gpus all -it --rm -v /data:/data ssds:local