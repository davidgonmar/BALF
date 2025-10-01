# BALF: Budgeted Activation-Aware Low-Rank Factorization for Fine-Tuning-Free Compression

This repository contains the code to reproduce the experiments for our paper.

*Author: David González Martínez*

*Paper: https://arxiv.org/abs/2509.25136*

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

The experiments need the CIFAR-10 dataset (train and eval), the CIFAR-10-C (eval) dataset, a subset of the ImageNet training set (to use as calibration data), and the ImageNet validation set (eval).

The CIFAR-10 dataset is automatically handled by Pytorch.

For the CIFAR-10-C dataset, download it from [here](https://zenodo.org/record/2535967#.Yk1n6HZBzDI) (at the time of writing) and extract it to `./CIFAR-10-C`.

For ImageNet, we provide the scripts we used to extract everything. You need to download the validation and training sets, as well as the dev-kit from [here](http://www.image-net.org/download). Then, use `./scripts/imagenet/extract_val.py` and `./scripts/imagenet/extract_calib.py` with the appropriate parameters to extract the sets. These will be placed in `./imagenet-val` and `./imagenet-calib` respectively.

The experiment scripts will pick those paths automatically.


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

Most scripts usage can be identified from the name. They generally also include a header with a brief description of their purpose.

### Results of Other Works
For comparison with other works, we either extracted the data from plots in the corresponding papers, or used data provided in tables. As we value transparency, we provide additional information in `./other-works-results.md`.

## Others

If you want to cite this work, you can use

```bibtex
@article{gonzalezmartinez2025balf,
  title         = {{BALF}: Budgeted Activation-Aware Low-Rank Factorization for Fine-Tuning-Free Model Compression,
  author        = {{Gonz{\'a}lez Mart{\'{\i}}nez}, David},
  year          = {2025},
  eprint        = {2509.25136},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  doi           = {10.48550/arXiv.2509.25136},
  url           = {https://arxiv.org/abs/2509.25136},
  note          = {arXiv preprint}
}
```

If you have any questions or suggestions, feel free to email me.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
