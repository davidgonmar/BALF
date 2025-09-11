#!/usr/bin/env python3
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
)
from lib.factorization.factorize import (
    to_low_rank_activation_aware_auto,
    collect_activation_cache,
)
from lib.utils.layer_fusion import (
    fuse_batch_norm_inference,
    fuse_conv_bn,
    get_conv_bn_fuse_pairs,
)

# -----------------------
# Args
# -----------------------
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
    ],
)
parser.add_argument("--results_dir", required=True)
parser.add_argument("--mode", required=True, choices=["flops_auto", "params_auto"])
parser.add_argument("--ratio", type=float, required=True, help="Fixed ratio_to_keep")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--train_dir", required=True)
parser.add_argument("--val_dir", required=True)
parser.add_argument("--batch_size_eval", type=int, default=256)
parser.add_argument("--batch_size_cache", type=int, default=128)
parser.add_argument(
    "--force_recache",
    action="store_true",
    help="Ignore any on-disk caches and recalc activations",
)
args = parser.parse_args()

seed_everything(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Model
# -----------------------
model_dict = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "mobilenet_v2": models.mobilenet_v2,
    "resnext50_32x4d": models.resnext50_32x4d,
    "resnext101_32x8d": models.resnext101_32x8d,
    "vit_b_16": models.vit_b_16,
}
model = model_dict[args.model_name](pretrained=True).to(device)
model.eval()

# -----------------------
# Data
# -----------------------
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

eval_tf = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ]
)
train_tf = eval_tf  # same normalization/resize

# Eval set (optionally sub-sampled for speed)
eval_ds_full = datasets.ImageFolder(args.val_dir, transform=eval_tf)
# Subset eval to a stable 3000 samples for fair comparisons
g_eval = torch.Generator().manual_seed(args.seed + 12345)
perm_eval = torch.randperm(len(eval_ds_full), generator=g_eval)[
    : min(3000, len(eval_ds_full))
]
eval_ds = Subset(eval_ds_full, perm_eval.tolist())

eval_dl = DataLoader(
    eval_ds,
    batch_size=args.batch_size_eval,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
)

# -----------------------
# Baseline (original)
# -----------------------
# Fuse Conv+BN for evaluation baseline where applicable
fuse_pairs = get_conv_bn_fuse_pairs(model)
model_fused = fuse_conv_bn(
    model, fuse_pairs, fuse_impl=fuse_batch_norm_inference, inplace=False
).to(device)
model_fused.eval()

baseline_metrics = evaluate_vision_model(model_fused, eval_dl)
params_orig = sum(p.numel() for p in model_fused.parameters())
flops_orig = count_model_flops(model_fused, (1, 3, 224, 224), formatted=False)

print(
    f"[original] loss={baseline_metrics['loss']:.4f} "
    f"acc={baseline_metrics['accuracy']:.4f} "
    f"params={params_orig} flops_total={flops_orig['total']}"
)

# -----------------------
# Experiment config
# -----------------------
# Hardcoded calibration sizes to sweep
CALIB_SIZES = [256, 512, 1024, 2048, 4096]

results_dir = Path(args.results_dir)
results_dir.mkdir(parents=True, exist_ok=True)

layer_keys = [k for k in get_all_convs_and_linears(model)]

metric_name = "flops" if args.mode == "flops_auto" else "params"

all_results = []

# -----------------------
# Sweep over calibration sizes
# -----------------------
train_ds_full = datasets.ImageFolder(args.train_dir, transform=train_tf)

# remove cache dir
cache_base_dir = results_dir / "whitening-calib-sweep" / args.model_name
if cache_base_dir.exists():
    import shutil

    shutil.rmtree(cache_base_dir)

for calib_size in CALIB_SIZES:
    actual_size = min(calib_size, len(train_ds_full))
    # Stable subset per size
    g_train = torch.Generator().manual_seed(args.seed + calib_size)
    idx = torch.randperm(len(train_ds_full), generator=g_train)[:actual_size].tolist()
    train_ds = Subset(train_ds_full, idx)

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size_cache,
        shuffle=True,
        drop_last=True if len(train_ds) >= args.batch_size_cache else False,
        num_workers=8,
        pin_memory=True,
    )

    activation_cache = collect_activation_cache(model, train_dl, keys=layer_keys)

    # Low-rank conversion (activation-aware auto) at fixed ratio
    model_lr = to_low_rank_activation_aware_auto(
        model,
        activation_cache,
        ratio_to_keep=args.ratio,
        inplace=False,
        keys=layer_keys,
        metric=metric_name,
        save_dir=str(cache_base_dir),
    )

    # remove save_dir recursively
    import shutil

    shutil.rmtree(str(cache_base_dir))

    # Fuse for eval
    model_eval = fuse_conv_bn(
        model_lr, fuse_pairs, fuse_impl=fuse_batch_norm_inference, inplace=False
    ).to(device)
    model_eval.eval()

    # Compute metrics
    params_lr = sum(p.numel() for p in model_eval.parameters())
    flops_lr = count_model_flops(model_eval, (1, 3, 224, 224), formatted=False)
    eval_lr = evaluate_vision_model(model_eval, eval_dl)

    params_ratio = float(params_lr / params_orig)
    flops_ratio = float(flops_lr["total"] / flops_orig["total"])

    print(
        f"[calib={actual_size} ratio={args.ratio:.6f} {args.mode}] "
        f"loss={eval_lr['loss']:.4f} acc={eval_lr['accuracy']:.4f} "
        f"params_ratio={params_ratio:.4f} flops_ratio={flops_ratio:.4f}"
    )

    # Save per-size metrics
    size_dir = results_dir / args.model_name / args.mode / f"calib_{actual_size}"
    size_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = size_dir / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump(
            {
                "calib_size": int(actual_size),
                "ratio": float(args.ratio),
                "mode": args.mode,
                "model_name": args.model_name,
                "seed": args.seed,
                "loss": float(eval_lr["loss"]),
                "accuracy": float(eval_lr["accuracy"]),
                "params_ratio": params_ratio,
                "flops_ratio": flops_ratio,
                "params_orig": int(params_orig),
                "flops_orig_total": int(flops_orig["total"]),
            },
            f,
            indent=2,
        )

    all_results.append(
        {
            "calib_size": int(actual_size),
            "ratio": float(args.ratio),
            "mode": args.mode,
            "loss": float(eval_lr["loss"]),
            "accuracy": float(eval_lr["accuracy"]),
            "params_ratio": params_ratio,
            "flops_ratio": flops_ratio,
        }
    )

# -----------------------
# Aggregate results
# -----------------------
with (results_dir / "results.json").open("w") as f:
    json.dump(all_results, f, indent=2)
