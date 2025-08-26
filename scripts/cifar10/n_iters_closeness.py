import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
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
    to_low_rank_activation_aware_auto,  # factorize model
    collect_activation_cache,  # use cached activations
)
from lib.utils.layer_fusion import (
    fuse_batch_norm_inference,
    fuse_conv_bn,
    get_conv_bn_fuse_pairs,
)
from lib.models import load_model

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--model_name", required=True, choices=["resnet20", "resnet56"])
parser.add_argument("--pretrained_path", required=True)
parser.add_argument("--results_dir", required=True)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
seed_everything(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = load_model(
    args.model_name,
).to(device)

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(cifar10_mean, cifar10_std)]
)

eval_ds = datasets.CIFAR10(root="data", train=False, transform=transform, download=True)
eval_dl = DataLoader(eval_ds, batch_size=512, shuffle=False)

train_ds = datasets.CIFAR10(root="data", train=True, transform=transform, download=True)
subset = torch.utils.data.Subset(train_ds, torch.randint(0, len(train_ds), (1024,)))
train_dl = DataLoader(subset, batch_size=256, shuffle=True, drop_last=True)

fuse_pairs = get_conv_bn_fuse_pairs(model)

model_fused = fuse_conv_bn(
    model, fuse_pairs, fuse_impl=fuse_batch_norm_inference, inplace=False
)

baseline_metrics = evaluate_vision_model(model_fused, eval_dl)
params_orig = sum(p.numel() for p in model_fused.parameters())
flops_orig = count_model_flops(model_fused, (1, 3, 32, 32), formatted=False)

print(
    f"[original] loss={baseline_metrics['loss']:.4f} acc={baseline_metrics['accuracy']:.4f} "
    f"params={params_orig} flops_total={flops_orig['total']}"
)
layer_keys = [k for k in get_all_convs_and_linears(model)]

# Build activation cache once, then reuse in the loop ---
activation_cache = collect_activation_cache(model, train_dl, keys=layer_keys)

base_dir = Path(args.results_dir)
base_dir.mkdir(parents=True, exist_ok=True)

results = []

n_iters_try = list(range(50))

for k in n_iters_try:
    model_lr = to_low_rank_activation_aware_auto(
        model,
        activation_cache,  # pass the cache instead of the dataloader
        ratio_to_keep=0.6,
        inplace=False,
        keys=layer_keys,
        metric="flops",
        n_iters=k,
    )
    model_eval = fuse_conv_bn(
        model_lr, fuse_pairs, fuse_impl=fuse_batch_norm_inference, inplace=False
    )
    params_lr = sum(p.numel() for p in model_eval.parameters())
    flops_raw_lr = count_model_flops(model_eval, (1, 3, 32, 32), formatted=False)
    eval_lr = evaluate_vision_model(model_eval.to(device), eval_dl)

    print(
        f"[ratio={k:.6f}] loss={eval_lr['loss']:.4f} acc={eval_lr['accuracy']:.4f} "
        f"params_ratio={params_lr/params_orig:.4f} flops_ratio={flops_raw_lr['total']/flops_orig['total']:.4f}"
    )

    print(f"delta params_ratio: {abs((params_lr/params_orig) - 0.6)}")
    print(
        f"delta flops_ratio: {abs((flops_raw_lr['total']/flops_orig['total']) - 0.6)}"
    )

    results.append(
        {
            "metric_value": k,
            "loss": float(eval_lr["loss"]),
            "accuracy": float(eval_lr["accuracy"]),
            "params_ratio": float(params_lr / params_orig),
            "flops_ratio": float(flops_raw_lr["total"] / flops_orig["total"]),
            "try_n_iters": k,
            "delta_flops_ratio": abs(
                (flops_raw_lr["total"] / flops_orig["total"]) - 0.6
            ),
            "delta_params_ratio": abs((params_lr / params_orig) - 0.6),
        }
    )

output_file = base_dir / "results.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)
