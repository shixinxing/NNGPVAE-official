# Nearest Neighbour Gaussian Process Variational AutoEncoder

This repository contains the official PyTorch implementation for the ICML 2025 paper "Neighbour-Driven Gaussian Process Variational Autoencoders (GPVAE) for Scalable Structured Latent Modelling." *(arxiv link TBD)* Inspired by recent developments in nearest neighbour Gaussian Process (GP) (i.e., [SWS-GP](https://proceedings.mlr.press/v139/tran21a.html) and [VNN-GP](https://proceedings.mlr.press/v162/wu22h.html)), this work introduces a neighbour-driven approximation strategy that exploits local adjacencies in the latent space to achieve scalable GPVAE inference.

<p align="center">
<img src="https://github.com/shixinxing/NNGPVAE-official/blob/main/assets/illustration-github.png" alt="示例图片" width="600">
</p>

## Table of Contents
- [Environment Setup](#environment-setup)
- [Usage](#usage)
- [Baseline Implementations](#baseline-implementations)
- [Citation](#citation)

## Environment Setup

## Usage

## Baseline Implementations

Below is a reference list of official GitHub repositories that implement GPVAE-related models:
| Baseline Model | Reference |
| ----- | -----|
| GPVAE-Casale [(code)](https://github.com/fpcasale/GPPVAE)  | Gaussian Process Prior Variational Autoencoders (NIPS 2018) [(paper)](https://arxiv.org/abs/1810.11738) |
| GPVAE-Pearce [(code)](https://github.com/scrambledpie/GPVAE) | The Gaussian Process Prior VAE for Interpretable Latent Dynamics from Pixels (AABI 2019) [(paper)](https://proceedings.mlr.press/v118/pearce20a.html) |
| GPVAE-Band [(code)](https://github.com/ratschlab/GP-VAE) | GP-VAE: Deep Probabilistic Time Series Imputation (AISTATS 2020) [(paper)](https://arxiv.org/abs/1907.04155) |
| SVGPVAE [(code)](https://github.com/ratschlab/SVGP-VAE) | Scalable Gaussian Process Variational Autoencoder (AISTATS 2021) [(paper)](https://arxiv.org/abs/2010.13472) |
| LVAE [(code)](https://github.com/SidRama/Longitudinal-VAE) | Longitudinal Variational Autoencoder (AISTATS 2021) [(paper)](https://proceedings.mlr.press/v130/ramchandran21b.html) |
| MGPVAE [(code)](https://github.com/harrisonzhu508/MGPVAE?tab=readme-ov-file) | Markovian Gaussian Process Variational Autoencoders (ICML 2023) [(paper)](https://arxiv.org/pdf/2207.05543) |
|SGPBAE [(code)](https://github.com/tranbahien/sgp-bae)| Fully Bayesian Autoencoders with Latent Sparse Gaussian Processes (ICML 2023) [(paper)](https://proceedings.mlr.press/v202/tran23a.html) |


## Citation

If you find this work helpful, please consider citing our ICML paper:

```
@inproceedings{nngpvae2025,
  title = {Neighbour-driven {G}aussian process variational autoencoders for scalable structured latent modelling},
  author = {Shi, Xinxing and Jiang, Xiaoyu and {\'A}lvarez, Mauricio},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning (ICML)},
  pages = {--},
  year = {2025},
  organization={PMLR}
}
```

