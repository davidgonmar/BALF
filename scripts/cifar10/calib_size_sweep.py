# sweep_cifar10.py
import argparse
import json
from pathlib import Path
import shutil

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from lib.utils import (
    cifar10_mean,
    cifar10_std,
    evaluate_vision_model,
    seed_everything,
    count_model_flops,
    get_all_convs_and_linears,
)
from lib.factorization.factorize import (
    to_low_rank_activation_aware_auto,
    collect_activation_cache,
)
from lib.models import load_model


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--model_name", required=True, choices=["resnet20", "resnet56"])
parser.add_argument("--pretrained_path", required=True)
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
parser.add_argument("--batch_size_eval", type=int, default=512)
parser.add_argument("--batch_size_cache", type=int, default=256)
parser.add_argument("--eval_subset_size", type=int, default=-1)
parser.add_argument("--data_root", type=str, default="data")
args = parser.parse_args()

seed_everything(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------------------------------
# Model
# -------------------------------------------------------------------------
model = load_model(args.model_name, pretrained_path=args.pretrained_path).to(device)
model.eval()

# -------------------------------------------------------------------------
# Data
# -------------------------------------------------------------------------
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(cifar10_mean, cifar10_std)]
)

eval_ds_full = datasets.CIFAR10(
    root=args.data_root, train=False, transform=transform, download=True
)
if args.eval_subset_size > 0 and args.eval_subset_size < len(eval_ds_full):
    g_eval = torch.Generator().manual_seed(args.seed + 12345)
    idx = torch.randperm(len(eval_ds_full), generator=g_eval)[
        : args.eval_subset_size
    ].tolist()
    eval_ds = Subset(eval_ds_full, idx)
else:
    eval_ds = eval_ds_full

eval_dl = DataLoader(
    eval_ds,
    batch_size=args.batch_size_eval,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
)

baseline_metrics = evaluate_vision_model(model, eval_dl)
params_orig = sum(p.numel() for p in model.parameters())
flops_orig = count_model_flops(model, (1, 3, 32, 32), formatted=False)

print(
    f"[original] loss={baseline_metrics['loss']:.4f} "
    f"acc={baseline_metrics['accuracy']:.4f} "
    f"params={params_orig} flops_total={flops_orig['total']}"
)

train_ds_full = datasets.CIFAR10(
    root=args.data_root, train=True, transform=transform, download=True
)

CALIB_SIZES = [256, 512, 1024, 2048, 4096, 8192, 16384]

results_dir = Path(args.results_dir)
results_dir.mkdir(parents=True, exist_ok=True)

layer_keys = [k for k in get_all_convs_and_linears(model)]
metric_name = "flops" if args.mode == "flops_auto" else "params"

all_results = []

cache_base_dir = Path(
    f"./cifar10_calib_size_sweep/{args.model_name}/activation_caches/"
)

# -------------------------------------------------------------------------
# Sweep
# -------------------------------------------------------------------------
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
        shuffle=True,
        drop_last=True if len(train_ds) >= args.batch_size_cache else False,
        num_workers=8,
        pin_memory=True,
    )

    cache_file = cache_base_dir / f"activation_cache_calib_{calib_size}.pt"

    if cache_file.exists():
        activation_cache = torch.load(cache_file, map_location="cpu", weights_only=True)
    else:
        activation_cache = collect_activation_cache(model, train_dl, keys=layer_keys)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(activation_cache, cache_file)

    for ratio in args.ratios:
        model_lr = to_low_rank_activation_aware_auto(
            model,
            activation_cache,
            ratio_to_keep=ratio,
            inplace=False,
            keys=layer_keys,
            metric=metric_name,
            save_dir=str(cache_base_dir),
        )

        shutil.rmtree(str(cache_base_dir))

        model_eval = model_lr.to(device)
        model_eval.eval()

        params_lr = sum(p.numel() for p in model_eval.parameters())
        flops_lr = count_model_flops(model_eval, (1, 3, 32, 32), formatted=False)
        eval_lr = evaluate_vision_model(model_eval, eval_dl)

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
