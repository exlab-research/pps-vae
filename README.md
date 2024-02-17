# [Partial Pixel Space Variational Autoencoder (PPS-VAE)](https://arxiv.org/abs/2305.18485)
This repository contains an implementation of PPS-VAE model presented in the "Autoencoding Conditional Neural Processes for Representation Learning" paper.

<p align="center">
    <img src="/images/demonstrative_figure.png" width="400" height="400">
</p>

## PPS-VAE usage
To train a new PPS-VAE model:
```
$ cd ./src
```
then run:
```
python3 main.py --config_file ../configs/clevr/exp_3.ini 
```

## Datasets

1) Imagenet --- we use the huggingface (Maysee/tiny-imagenet)
2) celeba or fer2013 --- we download the datasets separately and place the datasets in the ./data
3) clever --- we use torchvision datasets



## Contact info

For questions or more information please use the following:
* **Email:** victorprokhorov91@gmail.com