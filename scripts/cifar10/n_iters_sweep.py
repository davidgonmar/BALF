"""
This script is used to evaluate how close our allocator gets to the budged based on
the number of iterations passed to it.
"""

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
    make_factorization_cache_location,
)
from lib.factorization.factorize import (
    to_low_rank_activation_aware_auto,
    collect_activation_cache,
)
from lib.models import load_model


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Repeatable flags: e.g. --model resnet20 --pretrained pathA --model resnet56 --pretrained pathB
    parser.add_argument(
        "--model",
        dest="models",
        action="append",
        required=True,
        choices=["resnet20", "resnet56"],
        help="Repeat to evaluate multiple models.",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained_paths",
        action="append",
        required=True,
        help="Repeat to pair with each --model.",
    )
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ratio_to_keep", type=float, default=0.6)
    parser.add_argument(
        "--iters_max",
        type=int,
        default=50,
        help="Max n_iters to try (inclusive 0..iters_max).",
    )
    parser.add_argument("--calib_size", type=int, default=1024)
    args = parser.parse_args()

    if len(args.models) != len(args.pretrained_paths):
        raise ValueError(
            f"Got {len(args.models)} --model but {len(args.pretrained_paths)} --pretrained. They must match 1:1."
        )
    return args


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Shared datasets/dataloaders (reused across models)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(cifar10_mean, cifar10_std)]
    )
    eval_ds = datasets.CIFAR10(
        root="data", train=False, transform=transform, download=True
    )
    eval_dl = DataLoader(eval_ds, batch_size=512)

    train_ds = datasets.CIFAR10(
        root="data", train=True, transform=transform, download=True
    )

    subset_indices = torch.randperm(len(train_ds))[: args.calib_size]
    subset = torch.utils.data.Subset(train_ds, subset_indices)
    train_dl = DataLoader(subset, batch_size=256)

    base_dir = Path(args.results_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    global_results = {}

    n_iters_try = list(range(1, args.iters_max + 1))

    for model_name, ckpt in zip(args.models, args.pretrained_paths):
        model = load_model(model_name, ckpt).to(device)

        baseline_metrics = evaluate_vision_model(model, eval_dl)
        params_orig = sum(p.numel() for p in model.parameters())
        flops_orig = count_model_flops(model, (1, 3, 32, 32))

        print(
            f"[{model_name} | original] loss={baseline_metrics['loss']:.4f} "
            f"acc={baseline_metrics['accuracy']:.4f} "
            f"params={params_orig} flops_total={flops_orig['total']}"
        )

        layer_keys = get_all_convs_and_linears(model)

        activation_cache = collect_activation_cache(model, train_dl, keys=layer_keys)

        results = []

        for k in n_iters_try:
            model_lr = to_low_rank_activation_aware_auto(
                model,
                activation_cache,  # cached activations
                ratio_to_keep=args.ratio_to_keep,
                inplace=False,
                keys=layer_keys,
                metric="flops",
                n_iters=k,
                save_dir=make_factorization_cache_location(
                    model_name,
                    args.calib_size,
                    "cifar10",
                    "n_iters_sweep",
                    args.seed,
                ),
            )
            params_lr = sum(p.numel() for p in model_lr.parameters())
            flops_raw_lr = count_model_flops(model_lr, (1, 3, 32, 32))
            eval_lr = evaluate_vision_model(model_lr, eval_dl)

            params_ratio = params_lr / params_orig
            flops_ratio = flops_raw_lr["total"] / flops_orig["total"]

            print(
                f"[{model_name} | iters={k}] loss={eval_lr['loss']:.4f} "
                f"acc={eval_lr['accuracy']:.4f} "
                f"params_ratio={params_ratio:.4f} flops_ratio={flops_ratio:.4f}"
            )
            print(f"delta flops_ratio: {abs(flops_ratio - args.ratio_to_keep)}")

            results.append(
                {
                    "model": model_name,
                    "metric_value": k,
                    "loss": float(eval_lr["loss"]),
                    "accuracy": float(eval_lr["accuracy"]),
                    "params_ratio": float(params_ratio),
                    "flops_ratio": float(flops_ratio),
                    "try_n_iters": k,
                    "delta_flops_ratio": float(abs(flops_ratio - args.ratio_to_keep)),
                    "baseline_loss": float(baseline_metrics["loss"]),
                    "baseline_accuracy": float(baseline_metrics["accuracy"]),
                    "params_orig": int(params_orig),
                    "flops_orig_total": int(flops_orig["total"]),
                    "ratio_to_keep": float(args.ratio_to_keep),
                }
            )

        global_results[model_name] = results

    # Top-level summary
    with open(base_dir / "summary.json", "w") as f:
        json.dump(global_results, f, indent=2)


if __name__ == "__main__":
    main()
