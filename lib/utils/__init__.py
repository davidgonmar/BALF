from typing import List, Callable, Dict
import torch
from torch import nn
import functools as ft
from .flops import count_model_flops
from .layer_fusion import fuse_batch_norm_inference
import random
import numpy as np
from .general import gather_submodules, keys_passlist_should_do, unzip, get_all_convs_and_linears, extract_weights, AverageMeter, seed_everything, replace_with_factory, is_linear, is_conv2d

def evaluate_vision_model(
    model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, eval=True
) -> Dict[str, float]:
    prev_state = model.training
    if eval:
        model.eval()
    else:
        model.train()

    device = next(model.parameters()).device

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
    model.train(prev_state)
    return {"accuracy": acc_meter.avg, "loss": loss_meter.avg}


cifar10_mean = [0.4914, 0.4822, 0.4465]
cifar10_std = [0.2470, 0.2435, 0.2616]