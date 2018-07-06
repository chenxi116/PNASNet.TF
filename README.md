# PNASNet.TF

TensorFlow implementation of [PNASNet-5](https://arxiv.org/abs/1712.00559). While completely compatible with the [official implementation](https://github.com/tensorflow/models/blob/master/research/slim/nets/nasnet/pnasnet.py), this implementation focuses on simplicity and inference.

## Requirements

- TensorFlow 1.8.0
- PyTorch 0.4.0
- torchvision 0.2.1

## Data and Model Preparation

- Download the ImageNet validation set and move images to labeled subfolders. Todo this, you can use [this script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh). Make sure the folder `val` is under `data/`.
- Download the `PNASNet-5_Large_331` pretrained model:
```bash
cd data
wget https://storage.googleapis.com/download.tensorflow.org/models/pnasnet-5_large_2017_12_13.tar.gz
tar xvf pnasnet-5_large_2017_12_13.tar.gz
```

## Usage

```bash
python main.py
```

The last printed line should read:
```bash
Test: [50000/50000]	Prec@1 0.829	Prec@5 0.962
```
