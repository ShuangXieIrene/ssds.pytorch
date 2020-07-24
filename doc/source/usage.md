## Usage
### 0. Check the config file by Visualization
Defined the network in a [config file](../experiments/cfgs/tests/test.yml) and tweak the config file based on the visualized anchor boxes
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
