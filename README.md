# zsl_methods_imagenet

Zero-shot learning methods for ImageNet:
- [CADA-VAE](https://arxiv.org/abs/1812.01784) (remplementation, see the [original implementation](https://github.com/edgarschnfld/cada-vae-pytorch)):
- SimpleZSL (based on [DeViSE](https://papers.nips.cc/paper/2013/hash/7cce53cf90577442771720a370c3c723-Abstract.html))

## Configs and models

Config files in "config" folders are just examples.
Config files and models used in our experiments are available to download:
[Config files and models](https://kth.box.com/s/coeguix1cyk2umf3ba0z16nju70dgx28)

## Conda environment

[conda.yml](./conda.yaml) contains a Conda environment used for the project.
Note that it contains more dependencies than this project requires!

## Setup

The project requires to have subprojects directories in `PYTHONPATH`:

```sh
export PYTHONPATH=$PYTHONPATH:${PROJECT_DIR}:${PROJECT_DIR}/CADA-VAE:${PROJECT_DIR}/SimpleZSL
```

## Project

The code from this repository was used in our work, see [our project page](https://bujwid.eu/p/zsl-imagenet-wiki).
