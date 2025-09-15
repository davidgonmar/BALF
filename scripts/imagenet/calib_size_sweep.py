"""
This script sweeps over calibration dataset sizes for activation-aware
low-rank factorization on ImageNet.
"""

import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import torchvision.models as models

from lib.utils import (
    evaluate_vision_model,
    seed_everything,
    count_model_flops,
    get_all_convs_and_linears,
    maybe_retrieve_activation_cache,
    make_factorization_cache_location,
    imagenet_mean,
    imagenet_std,
)
from lib.factorization.factorize import (
    to_low_rank_activation_aware_auto,
)
import functools
import timm

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--model_name",
    required=True,
    choices=[
        "resnet18",
        "resnet34",
        "resnet50",
        "mobilenet_v2",
        "resnext50_32x4d",
        "resnext101_32x8d",
        "vit_b_16",
        "deit_b_16",
    ],
)
parser.add_argument("--results_dir", required=True)
parser.add_argument("--mode", required=True, choices=["flops_auto", "params_auto"])
parser.add_argument(
    "--ratios",
    type=float,
    nargs="+",
    required=True,
    help="One or more ratio_to_keep values, e.g. --ratios 0.8 0.6 0.4",
)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--train_dir", required=True)
parser.add_argument("--val_dir", required=True)
parser.add_argument("--batch_size_eval", type=int, default=256)
parser.add_argument("--batch_size_cache", type=int, default=128)
parser.add_argument("--eval_subset_size", type=int, default=-1)

args = parser.parse_args()

seed_everything(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model_dict = {
    "resnet18": functools.partial(
        models.resnet18, weights=models.ResNet18_Weights.IMAGENET1K_V1
    ),
    "resnet50": functools.partial(
        models.resnet50, weights=models.ResNet50_Weights.IMAGENET1K_V1
    ),
    "mobilenet_v2": functools.partial(
        models.mobilenet_v2, weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
    ),
    "resnext50_32x4d": functools.partial(
        models.resnext50_32x4d, weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1
    ),
    "resnext101_32x8d": functools.partial(
        models.resnext101_32x8d, weights=models.ResNeXt101_32X8D_Weights.IMAGENET1K_V1
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
model = model_dict[args.model_name]().to(device)
model.eval()

interp_mode = (
    transforms.InterpolationMode.BILINEAR
    if args.model_name
    not in [
        "vit_b_16",
        "deit_b_16",
    ]
    else transforms.InterpolationMode.BICUBIC
)
eval_tf = transforms.Compose(
    [
        transforms.Resize(256, interpolation=interp_mode),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ]
)
train_tf = eval_tf

eval_ds_full = datasets.ImageFolder(args.val_dir, transform=eval_tf)

if args.eval_subset_size > 0 and args.eval_subset_size < len(eval_ds_full):
    g_eval = torch.Generator().manual_seed(args.seed + 12345)
    perm_eval = torch.randperm(len(eval_ds_full), generator=g_eval)[
        : args.eval_subset_size
    ]
    eval_ds = Subset(eval_ds_full, perm_eval.tolist())
else:
    eval_ds = eval_ds_full


eval_dl = DataLoader(
    eval_ds,
    batch_size=args.batch_size_eval,
    num_workers=8,
    pin_memory=True,
)

baseline_metrics = evaluate_vision_model(model, eval_dl)
params_orig = sum(p.numel() for p in model.parameters())
flops_orig = count_model_flops(model, (1, 3, 224, 224))

print(
    f"[original] loss={baseline_metrics['loss']:.4f} "
    f"acc={baseline_metrics['accuracy']:.4f} "
    f"params={params_orig} flops_total={flops_orig['total']}"
)

CALIB_SIZES = [256, 512, 1024, 2048, 4096, 8192, 16384]

results_dir = Path(args.results_dir)
results_dir.mkdir(parents=True, exist_ok=True)

layer_keys = [k for k in get_all_convs_and_linears(model)]

metric_name = "flops" if args.mode == "flops_auto" else "params"

all_results = []

train_ds_full = datasets.ImageFolder(args.train_dir, transform=train_tf)

for calib_size in CALIB_SIZES:
    assert calib_size <= len(
        train_ds_full
    ), f"calib_size {calib_size} > {len(train_ds_full)}"

    g_train = torch.Generator().manual_seed(args.seed + calib_size)
    idx = torch.randperm(len(train_ds_full), generator=g_train)[:calib_size].tolist()
    train_ds = Subset(train_ds_full, idx)

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size_cache,
        num_workers=8,
        pin_memory=True,
    )

    activation_cache = maybe_retrieve_activation_cache(
        args.model_name,
        calib_size,
        "imagenet",
        "calib_size_sweep",
        args.seed,
        model,
        train_dl,
        layer_keys,
    )

    for ratio in args.ratios:
        model_lr = to_low_rank_activation_aware_auto(
            model,
            activation_cache,
            ratio_to_keep=ratio,
            inplace=False,
            keys=layer_keys,
            metric=metric_name,
            save_dir=make_factorization_cache_location(
                args.model_name, calib_size, "imagenet", "calib_size_sweep", args.seed
            ),
        )
        model_lr.eval()

        params_lr = sum(p.numel() for p in model_lr.parameters())
        flops_lr = count_model_flops(model_lr, (1, 3, 224, 224))
        eval_lr = evaluate_vision_model(model_lr, eval_dl)

        params_ratio = float(params_lr / params_orig)
        flops_ratio = float(flops_lr["total"] / flops_orig["total"])

        print(
            f"[calib={calib_size} ratio={ratio:.6f} {args.mode}] "
            f"loss={eval_lr['loss']:.4f} acc={eval_lr['accuracy']:.4f} "
            f"params_ratio={params_ratio:.4f} flops_ratio={flops_ratio:.4f}"
        )

        all_results.append(
            {
                "calib_size": int(calib_size),
                "ratio": float(ratio),
                "mode": args.mode,
                "loss": float(eval_lr["loss"]),
                "accuracy": float(eval_lr["accuracy"]),
                "params_ratio": params_ratio,
                "flops_ratio": flops_ratio,
            }
        )

with (results_dir / "results.json").open("w") as f:
    json.dump(all_results, f, indent=2)
