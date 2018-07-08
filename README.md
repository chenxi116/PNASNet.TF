# PNASNet.TF

TensorFlow implementation of [PNASNet-5](https://arxiv.org/abs/1712.00559). While completely compatible with the [official implementation](https://github.com/tensorflow/models/blob/master/research/slim/nets/nasnet/pnasnet.py), this implementation focuses on simplicity and inference.

In particular, three files of 1200 lines in total (`nasnet.py`, `nasnet_utils.py`, `pnasnet.py`) are refactored into two files of 400 lines in total (`cell.py`, `pnasnet.py`). This code no longer supports `NCHW` data format, primarily because the released model was trained with `NHWC`. I tried to keep the rough structure and all functionalities of the official implementation when simplifying it.

If you use the code, please cite:
```bash
@inproceedings{liu2018progressive,
  author    = {Chenxi Liu and
               Barret Zoph and
               Maxim Neumann and
               Jonathon Shlens and
               Wei Hua and
               Li{-}Jia Li and
               Li Fei{-}Fei and
               Alan L. Yuille and
               Jonathan Huang and
               Kevin Murphy},
  title     = {Progressive Neural Architecture Search},
  booktitle = {European Conference on Computer Vision},
  year      = {2018}
}
```

## Requirements

- TensorFlow 1.8.0
- torchvision 0.2.1 (for dataset loading)

## Data and Model Preparation

- Download the ImageNet validation set and move images to labeled subfolders. To do the latter, you can use [this script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh). Make sure the folder `val` is under `data/`.
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
