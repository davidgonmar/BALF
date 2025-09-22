from typing import Callable, Dict
import functools as ft
import torch
from torch import nn
import random
import numpy as np


def get_all_convs_and_linears(model):
    res = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            res.append(name)
    return res


def gather_submodules(model: nn.Module, should_do: Callable) -> list:
    return [
        (name, module)
        for name, module in model.named_modules()
        if should_do(module, name)
    ]


def keys_passlist_should_do(keys):
    return ft.partial(lambda keys, module, full_name: full_name in keys, keys)


def replace_with_factory(
    model: nn.Module, module_dict: Dict[str, nn.Module], factory_fn: Callable
):
    for name, module in module_dict.items():
        parent_module = model
        *parent_path, attr_name = name.split(".")
        for part in parent_path:
            parent_module = getattr(parent_module, part)
        setattr(parent_module, attr_name, factory_fn(name, module))
        del module
        module_dict[name] = None
    return model


def is_conv2d(module: nn.Module) -> bool:
    return isinstance(module, (torch.nn.Conv2d, torch.nn.LazyConv2d))


def is_linear(module: nn.Module) -> bool:
    return isinstance(module, (torch.nn.Linear, torch.nn.LazyLinear))


def seed_everything(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0.0
