from typing import List, Callable, Dict
import torch
from torch import nn
import functools as ft
from .flops import count_model_flops
from .layer_fusion import fuse_batch_norm_inference
import random
import numpy as np
from .general import (
    gather_submodules,
    keys_passlist_should_do,
    unzip,
    get_all_convs_and_linears,
    extract_weights,
    AverageMeter,
    seed_everything,
    replace_with_factory,
    is_linear,
    is_conv2d,
)
from typing import Tuple, Union
from torch.utils.data import DataLoader, Subset
from pathlib import Path
import contextlib as cl


@torch.no_grad()
def evaluate_vision_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    eval=True,
    fp16=False,
) -> Dict[str, float]:
    prev_state = model.training
    if eval:
        model.eval()
    else:
        model.train()
    import time

    device = next(model.parameters()).device
    ctx = torch.amp.autocast(device_type=device.type, enabled=fp16, dtype=torch.float16)
    with ctx:
        timer_start = time.time()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        criterion = torch.nn.CrossEntropyLoss()
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            batch_size = labels.size(0)
            accuracy = correct / batch_size

            loss_meter.update(loss.item(), batch_size)
            acc_meter.update(accuracy, batch_size)

        timer_end = time.time()
        print(
            f"Eval time: {timer_end - timer_start:.2f}s over {len(dataloader.dataset)} samples"
        )
    model.train(prev_state)
    return {"accuracy": acc_meter.avg, "loss": loss_meter.avg}


cifar10_mean = [0.4914, 0.4822, 0.4465]
cifar10_std = [0.2470, 0.2435, 0.2616]

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]


def make_factorization_cache_location(
    model_name: str,
    calib_size: int,
    dataset_name: str,
    script_key: str,
    seed: int,
) -> str:
    # script_key is done so that different scripts do not share cache
    # this is safer for reproducibility as script execution order may vary
    # seed allows for different script runs with different seeds to not share cache
    return f"./factorization-cache/{script_key}/{dataset_name}/{model_name}/seed-{seed}/calib-{calib_size}/"


def make_activation_cache_location(
    model_name: str,
    calib_size: int,
    dataset_name: str,
    script_key: str,
    seed: int,
) -> str:
    # script_key is done so that different scripts do not share cache
    # this is safer for reproducibility as script execution order may vary
    # seed allows for different script runs with different seeds to not share cache
    return f"./activation-cache/{script_key}/{dataset_name}/{model_name}/seed-{seed}/calib-{calib_size}/cache.pth"


def maybe_retrieve_activation_cache(
    model_name: str,
    calib_size: int,
    dataset_name: str,
    script_key: str,
    seed: int,
    model,
    dataloader: torch.utils.data.DataLoader,
    keys: List[str],
) -> Dict[str, torch.Tensor]:
    cache_loc = make_activation_cache_location(
        model_name, calib_size, dataset_name, script_key, seed
    )
    cache_file = Path(cache_loc)
    from lib.factorization.factorize import (
        collect_activation_cache,
    )  # avoid circular import

    if cache_file.exists():
        activation_cache = torch.load(cache_file, map_location="cpu", weights_only=True)
    else:
        activation_cache = collect_activation_cache(model, dataloader, keys=keys)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(activation_cache, cache_file)
    return activation_cache
