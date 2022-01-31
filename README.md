# zsl_methods_imagenet

Zero-shot learning methods for ImageNet:
- [CADA-VAE](https://arxiv.org/abs/1812.01784) (remplementation, see the [original implementation](https://github.com/edgarschnfld/cada-vae-pytorch)):
- SimpleZSL (based on [DeViSE](https://papers.nips.cc/paper/2013/hash/7cce53cf90577442771720a370c3c723-Abstract.html))

## Configs and models

Config files in "config" folders are just examples.
Config files and models used in our experiments are available to download:
[Config files and models](https://kth-my.sharepoint.com/:f:/g/personal/bujwid_ug_kth_se/EjLfX-VHCsZOgZrO8F6SqDUB4yxaRsmflyY9jg_A47R84w?e=nk5D6A)

`aux_feats` files with encoded class auxiliary (text) features were from: [`aux_feats` pickle files (See _Encoded Wikipedia articles (extracted features)_)](https://github.com/sebastianbujwid/zsl_text_imagenet#download-encoded-text-from-wikipedia-articles)

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
