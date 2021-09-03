# GwNet

This is a Gravitational Waves processing tool created to find gravitational wave signals from binary black hole collisions. It was created for [G2Net Gravitational-Wave Detection](https://www.kaggle.com/c/g2net-gravitational-wave-detection/overview) competition.

--- 
[![Python](https://img.shields.io/badge/Python-3.6-yellow.svg)]()
[![GPLv3](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/Sunnesoft/g2net-challenge/blob/main/LICENSE)

# Installation

To install, you should do:

```
pip install --upgrade git+git://github.com/Sunnesoft/g2net-challenge.git
```

# Examples

All examples were designed to run in [Google Colab](https://colab.research.google.com/) environment. 
To keep results [Google Drive](https://drive.google.com/) is used.

1. [The Kaggle dataset preprocessing with whiten, bandpass and notch filters](https://github.com/Sunnesoft/g2net-challenge/blob/main/examples/gw_create_filtered.ipynb).
2. Create constant Q-transform spectrograms from filtered dataset:
   1. [Slow numpy realisation](https://github.com/Sunnesoft/g2net-challenge/blob/main/examples/gw_create_cqt.ipynb).
   2. [Faster GPU/TPU realisation](https://github.com/Sunnesoft/g2net-challenge/blob/main/examples/gw_create_cqt_gpu.ipynb).
3. Prepare dataset for NN classification:
   1. [Train dataset](https://github.com/Sunnesoft/g2net-challenge/blob/main/examples/gw_prepare_train_dataset.ipynb).
   2. [Test dataset](https://github.com/Sunnesoft/g2net-challenge/blob/main/examples/gw_prepare_test_dataset.ipynb).
4. [Train EfficientNetB0, Xception, LeNet models](https://github.com/Sunnesoft/g2net-challenge/blob/main/examples/gw_train_model.ipynb).
5. [Test EfficientNetB0, Xception, LeNet models](https://github.com/Sunnesoft/g2net-challenge/blob/main/examples/gw_test_model.ipynb).

---
Some procedures of this tool adopted from more useful [GWpy tool](https://github.com/gwpy/gwpy.git).