#!/usr/bin/env python3
import argparse
import json
import shutil
import functools
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import torchvision.models as tvm
import timm

from lib.utils import (
    imagenet_mean,
    imagenet_std,
    seed_everything,
    get_all_convs_and_linears,
)
from lib.factorization.factorize import to_low_rank_activation_aware_auto

SUPPORTED = [
    "resnet18",
    "resnet50",
    "mobilenet_v2",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "vit_b_16",
    "deit_b_16",
]


def build_model(n):
    d = {
        "resnet18": functools.partial(
            tvm.resnet18, weights=tvm.ResNet18_Weights.IMAGENET1K_V1
        ),
        "resnet50": functools.partial(
            tvm.resnet50, weights=tvm.ResNet50_Weights.IMAGENET1K_V1
        ),
        "mobilenet_v2": functools.partial(
            tvm.mobilenet_v2, weights=tvm.MobileNet_V2_Weights.IMAGENET1K_V1
        ),
        "resnext50_32x4d": functools.partial(
            tvm.resnext50_32x4d, weights=tvm.ResNeXt50_32X4D_Weights.IMAGENET1K_V1
        ),
        "resnext101_32x8d": functools.partial(
            tvm.resnext101_32x8d, weights=tvm.ResNeXt101_32X8D_Weights.IMAGENET1K_V1
        ),
        "vit_b_16": functools.partial(
            timm.create_model,
            model_name="vit_base_patch16_224",
            num_classes=1000,
            pretrained=True,
        ),
        "deit_b_16": functools.partial(
            timm.create_model,
            model_name="deit_base_patch16_224",
            num_classes=1000,
            pretrained=True,
        ),
    }
    return d[n]()


def build_dl(train_dir, model_name, bs, calib_size, seed):
    interp = (
        transforms.InterpolationMode.BICUBIC
        if model_name in ["vit_b_16", "deit_b_16"]
        else transforms.InterpolationMode.BILINEAR
    )
    tf = transforms.Compose(
        [
            transforms.Resize(256, interpolation=interp),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )
    ds_full = datasets.ImageFolder(train_dir, transform=tf)
    if 0 < calib_size < len(ds_full):
        g = torch.Generator().manual_seed(seed + 12345)
        idx = torch.randperm(len(ds_full), generator=g)[:calib_size]
        ds = Subset(ds_full, idx.tolist())
    else:
        ds = ds_full
    return DataLoader(ds, batch_size=bs, num_workers=8, pin_memory=True, shuffle=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_dir", required=True)
    p.add_argument("--out", default="timings.json")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--calib_size", type=int, default=8192)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda")

    results = []
    tmp_root = Path(args.out).parent / "tmp_factor_cache"
    tmp_root.mkdir(parents=True, exist_ok=True)

    for name in SUPPORTED:
        print(f"Processing {name} ...")
        model = build_model(name).to(device)
        dl = build_dl(args.train_dir, name, args.batch_size, args.calib_size, args.seed)
        keys = list(get_all_convs_and_linears(model))
        save_dir = tmp_root / name
        save_dir.mkdir(parents=True, exist_ok=True)

        model_lr, timings = to_low_rank_activation_aware_auto(
            model=model,
            data_or_cache=dl,  # pass DataLoader to include cache time
            keys=keys,
            ratio_to_keep=0.6,  # fixed FLOPs keep
            metric="flops",
            inplace=True,
            save_dir=save_dir,  # use then delete
            benchmark=True,
        )

        shutil.rmtree(save_dir, ignore_errors=True)

        timings["model"] = name
        results.append(timings)

        # delete everything and clean cache
        del model
        del model_lr
        torch.cuda.empty_cache()

        # make sure memory allocated is less than 10MB
        assert (
            torch.cuda.memory_allocated() <= 10 * 1024 * 1024
        ), f"cuda memory should be less than 10MB but got {torch.cuda.memory_allocated()}"

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
