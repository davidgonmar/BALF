"""
This script evaluates low-rank factorization of ImageNet models.
It sweeps over different compression ratios and records accuracy.
It can use BALF (our proposed method) or other baselines.
"""

import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from lib.utils import (
    evaluate_vision_model,
    seed_everything,
    count_model_flops,
    get_all_convs_and_linears,
    imagenet_mean,
    imagenet_std,
    maybe_retrieve_activation_cache,
    make_factorization_cache_location,
)
from lib.factorization.factorize import (
    to_low_rank_activation_aware_auto,
    to_low_rank_activation_aware_manual,
    to_low_rank_manual,
)
from scripts.imagenet.factorize_sweep_values import get_values_for_model_and_mode
import torchvision.models as models
import timm
import functools


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--model_name",
    required=True,
    choices=[
        "resnet18",
        "resnet50",
        "mobilenet_v2",
        "resnext50_32x4d",
        "resnext101_32x8d",
        "vit_b_16",
        "deit_b_16",
    ],
)
parser.add_argument("--results_dir", required=True)
parser.add_argument(
    "--mode",
    default="flops_auto",
    choices=[
        "flops_auto",
        "params_auto",
        "energy_act_aware",
        "energy",
        "uniform",
        "uniform_act_aware",
    ],
)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--train_dir", required=True)
parser.add_argument("--val_dir", required=True)
parser.add_argument("--batch_size_eval", type=int, default=256)
parser.add_argument("--batch_size_cache", type=int, default=128)
parser.add_argument("--calib_size", type=int, default=8192)
parser.add_argument("--save_compressed_models", action="store_true")
parser.add_argument(
    "--eval_subset_size",
    type=int,
    default=-1,
    help="If >0, use a random subset of this many samples from val for eval",
)
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

# We use the recommended preprocessing for each model (based on timm for the vision transformers)
# See also https://huggingface.co/spaces/Roll20/pet_score/blob/b258ef28152ab0d5b377d9142a23346f863c1526/lib/timm/data/transforms_factory.py for computing the resize_size
interp_mode = (
    transforms.InterpolationMode.BICUBIC
    if args.model_name in ["vit_b_16", "deit_b_16"]
    else transforms.InterpolationMode.BILINEAR
)
ds_mean, ds_std = (
    ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    if args.model_name == "vit_b_16"
    else (imagenet_mean, imagenet_std)
)
resize = 248 if args.model_name in ["vit_b_16", "deit_b_16"] else 256

eval_tf = transforms.Compose(
    [
        transforms.Resize(resize, interpolation=interp_mode),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=ds_mean, std=ds_std),
    ]
)
train_tf = transforms.Compose(
    [
        transforms.Resize(resize, interpolation=interp_mode),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=ds_mean, std=ds_std),
    ]
)

eval_ds = datasets.ImageFolder(args.val_dir, transform=eval_tf)

if args.eval_subset_size > 0 and args.eval_subset_size < len(eval_ds):
    eval_g = torch.Generator().manual_seed(args.seed + 12345)
    eval_perm = torch.randperm(len(eval_ds), generator=eval_g)[: args.eval_subset_size]
    eval_ds = Subset(eval_ds, eval_perm.tolist())

eval_dl = DataLoader(
    eval_ds,
    batch_size=args.batch_size_eval,
    num_workers=8,
    pin_memory=True,
)

train_ds_full = datasets.ImageFolder(args.train_dir, transform=train_tf)

if args.calib_size > 0 and args.calib_size < len(train_ds_full):
    train_g = torch.Generator().manual_seed(args.seed + 12345)
    train_perm = torch.randperm(len(train_ds_full), generator=train_g)[
        : args.calib_size
    ]
    train_ds = Subset(train_ds_full, train_perm.tolist())
else:
    train_ds = train_ds_full
train_dl = DataLoader(
    train_ds,
    batch_size=args.batch_size_cache,
    num_workers=8,
    pin_memory=True,
)

baseline_metrics = evaluate_vision_model(model, eval_dl)
params_orig = sum(p.numel() for p in model.parameters())
flops_orig = count_model_flops(model, (1, 3, 224, 224))

print(
    f"[original] loss={baseline_metrics['loss']:.4f} acc={baseline_metrics['accuracy']:.4f} "
    f"params={params_orig} flops_total={flops_orig['total']}"
)


layer_keys = [k for k in get_all_convs_and_linears(model)]

activation_cache = maybe_retrieve_activation_cache(
    args.model_name,
    args.calib_size,
    "imagenet",
    "factorize_sweep",
    args.seed,
    model,
    train_dl,
    layer_keys,
)
results = []

for k in get_values_for_model_and_mode(args.model_name, args.mode):
    if args.mode == "flops_auto" or args.mode == "params_auto":
        model_lr = to_low_rank_activation_aware_auto(
            model,
            activation_cache,
            ratio_to_keep=k,
            inplace=False,
            keys=layer_keys,
            metric="flops" if args.mode == "flops_auto" else "params",
            save_dir=make_factorization_cache_location(
                args.model_name,
                args.calib_size,
                "imagenet",
                "factorize_sweep",
                args.seed,
            ),
        )
    elif args.mode == "energy_act_aware" or args.mode == "uniform_act_aware":
        name = (
            "svals_energy_ratio_to_keep"
            if args.mode == "energy_act_aware"
            else "uniform_compression_ratio_to_keep"
        )

        model_lr = to_low_rank_activation_aware_manual(
            model,
            activation_cache,
            cfg_dict={kk: {"name": name, "value": k} for kk in layer_keys},
            inplace=False,
            save_dir=make_factorization_cache_location(
                args.model_name,
                args.calib_size,
                "imagenet",
                "factorize_sweep",
                args.seed,
            ),
        )
    elif args.mode == "energy" or args.mode == "uniform":
        name = (
            "svals_energy_ratio_to_keep"
            if args.mode == "energy"
            else "uniform_compression_ratio_to_keep"
        )
        model_lr = to_low_rank_manual(
            model,
            cfg_dict={kk: {"name": name, "value": k} for kk in layer_keys},
            inplace=False,
        )
    params_lr = sum(p.numel() for p in model_lr.parameters())
    flops_raw_lr = count_model_flops(model_lr, (1, 3, 224, 224))
    eval_lr = evaluate_vision_model(model_lr.to(device), eval_dl)

    print(
        f"[ratio={k:.6f}] loss={eval_lr['loss']:.4f} acc={eval_lr['accuracy']:.4f} "
        f"params_ratio={params_lr/params_orig:.4f} flops_ratio={flops_raw_lr['total']/flops_orig['total']:.4f}"
    )

    results.append(
        {
            "metric_value": k,
            "loss": float(eval_lr["loss"]),
            "accuracy": float(eval_lr["accuracy"]),
            "params_ratio": float(params_lr / params_orig),
            "flops_ratio": float(flops_raw_lr["total"] / flops_orig["total"]),
            "mode": args.mode,
        }
    )

    # save model in /models, mainly to debug
    if args.save_compressed_models:
        model_path = (
            Path(args.results_dir)
            / "models"
            / f"{args.model_name}_mode{args.mode}_ratio{k:.5f}.pth"
        )
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model_lr, model_path)


results.append(
    {
        "metric_value": "original",
        "loss": float(baseline_metrics["loss"]),
        "accuracy": float(baseline_metrics["accuracy"]),
        "params_ratio": 1.0,
        "flops_ratio": 1.0,
        "mode": args.mode,
    }
)
output_file = Path(args.results_dir) / "results.json"
Path(args.results_dir).mkdir(parents=True, exist_ok=True)
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)
