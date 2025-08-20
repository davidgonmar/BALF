## Reproducing

The scripts to reproduce the experiments can be found in `./scripts`.

Make sure to run
```
export PYTHONPATH=.
```
before running the scripts.


### CIFAR Models Pre-Training

For ImageNet experiments, we use publicly available checkpoints. For CIFAR-10 models, we train our own. In order to train the models, use
```
./scripts/train_cifar.sh
```
