# BALF: Budgeted Activation-Aware Low-Rank Factorization for Fine-Tuning-Free Compression

This repository contains the code to reproduce the experiments for our paper.

The `./lib` directory contains the implementation of the generalized activation-aware factorization method, as well as the rank allocation strategy.
It also allows for energy allocation and selecting fixed rank ratios per layer, as well as regular (non-activation-aware) SVD-based factorization (energy-based).

It includes both the methods to transform a model into its factorized counterpart, as well as the factorized layers.


## Reproducing Experiments

### Installing Dependencies

Our experiment code was run on Python 3.10 and Python 3.11.

The required packages and their versions can be installed with
```bash
pip install -r requirements.txt
```


### Preparing the Data

The experiments need CIFAR-10, CIFAR-10-C, ImageNet, ImageNet-C, and ImageNet-V2. Dataset locations are configured independently through environment variables; there is no common data-root setting.

Example dataset paths are:

```bash
export CIFAR10_ROOT=/path/to/cifar10
export CIFAR10C_ROOT=/path/to/CIFAR-10-C
export IMAGENET_ROOT=/path/to/imagenet
export IMAGENETC_ROOT=/path/to/ImageNet-C
export IMAGENETV2_ROOT=/path/to/ImageNet-V2
```

CIFAR-10 is downloaded automatically by torchvision under `CIFAR10_ROOT`. Download and extract CIFAR-10-C, ImageNet-C, and ImageNet-V2 into their respective roots before launching experiments. ImageNet must use the standard torchvision `ImageFolder` layout under `${IMAGENET_ROOT}/train` and `${IMAGENET_ROOT}/val`. The defaults can be overridden individually for another machine or cluster.


### CIFAR Models Pre-Training

For ImageNet experiments, we use publicly available checkpoints. For CIFAR-10 models, we train our own following standard recipes. In order to train the models, use
```bash
./scripts/cifar10/pretrain_resnet.sh
```
It will train both the ResNet-20 and ResNet-56 models.


### Running the Experiments

Make sure to run
```
export PYTHONPATH=.
```
before running the scripts.

The scripts to reproduce the experiments can be found in `./scripts`. In general, for all experiments, you will find a ``.sh`` script that calls the corresponding Python script with the appropriate parameters. Those are the ones used to obtain the results in the paper. 

A lot of experiments cache activations and SVD artifacts so that they do not need to be recomputed every time. These are stored in `./activation-cache` and `./factorization-cache` respectively. You can delete those folders (or the ones specific to the script or model you want to run) if you want to recompute everything from scratch.

Results are printed to the console and also saved in a text file in `./results`. Those include raw data in the form of json files, as well as plots (in pdf format) used in the paper and tabular data in LaTeX format.

Most scripts usage can be identified from the name. They generally also include a header with a brief description of their purpose. Reusable artifacts for the duration of a run are stored under `/tmp`; set `BALF_CACHE_ROOT` to override that location.

### Results of Other Works
For comparison with other works, we either extracted the data from plots in the corresponding papers, or used data provided in tables. As we value transparency, we provide additional information in `./other-works-results.md`.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
