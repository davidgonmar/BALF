import argparse
import json
import functools
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import torchvision.models as models
import timm

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
from lib.factorization.factorize import to_low_rank_activation_aware_auto


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument(
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
    p.add_argument("--results_dir", required=True)
    p.add_argument("--train_dir", required=True)
    p.add_argument("--val_dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch_size_eval", type=int, default=256)
    p.add_argument("--batch_size_cache", type=int, default=128)
    p.add_argument("--calib_size", type=int, default=4096)
    p.add_argument("--eval_subset_size", type=int, default=-1)
    p.add_argument("--ratio_to_keep", type=float, default=0.6)
    p.add_argument(
        "--iters_max", type=int, default=49, help="Inclusive sweep 0..iters_max"
    )
    return p.parse_args()


def build_model(name):
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
            models.resnext101_32x8d,
            weights=models.ResNeXt101_32X8D_Weights.IMAGENET1K_V1,
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
    return model_dict[name]()


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(args.model_name).to(device)

    interp_mode = (
        transforms.InterpolationMode.BICUBIC
        if args.model_name in ["vit_b_16", "deit_b_16"]
        else transforms.InterpolationMode.BILINEAR
    )
    tf = transforms.Compose(
        [
            transforms.Resize(256, interpolation=interp_mode),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )

    eval_ds_full = datasets.ImageFolder(args.val_dir, transform=tf)
    if 0 < args.eval_subset_size < len(eval_ds_full):
        g = torch.Generator().manual_seed(args.seed + 12345)
        idx = torch.randperm(len(eval_ds_full), generator=g)[: args.eval_subset_size]
        eval_ds = Subset(eval_ds_full, idx.tolist())
    else:
        eval_ds = eval_ds_full
    eval_dl = DataLoader(
        eval_ds, batch_size=args.batch_size_eval, num_workers=8, pin_memory=True
    )

    train_ds_full = datasets.ImageFolder(args.train_dir, transform=tf)
    if 0 < args.calib_size < len(train_ds_full):
        g = torch.Generator().manual_seed(args.seed + 54321)
        idx = torch.randperm(len(train_ds_full), generator=g)[: args.calib_size]
        train_ds = Subset(train_ds_full, idx.tolist())
    else:
        train_ds = train_ds_full
    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size_cache, num_workers=8, pin_memory=True
    )

    baseline = evaluate_vision_model(model, eval_dl)
    params_orig = sum(p.numel() for p in model.parameters())
    flops_orig = count_model_flops(model, (1, 3, 224, 224))

    print(
        f"[{args.model_name} | original] loss={baseline['loss']:.4f} "
        f"acc={baseline['accuracy']:.4f} params={params_orig} flops_total={flops_orig['total']}"
    )

    layer_keys = [k for k in get_all_convs_and_linears(model)]
    activation_cache = maybe_retrieve_activation_cache(
        args.model_name,
        args.calib_size,
        "imagenet",
        "n_iters_sweep",
        args.seed,
        model,
        train_dl,
        layer_keys,
    )

    results = []
    for k in range(1, args.iters_max + 1, 10):
        model_lr = to_low_rank_activation_aware_auto(
            model,
            activation_cache,
            ratio_to_keep=args.ratio_to_keep,
            inplace=False,
            keys=layer_keys,
            metric="flops",
            n_iters=k,
            save_dir=make_factorization_cache_location(
                args.model_name, args.calib_size, "imagenet", "n_iters_sweep", args.seed
            ),
        )
        params_lr = sum(p.numel() for p in model_lr.parameters())
        flops_lr = count_model_flops(model_lr, (1, 3, 224, 224))
        eval_lr = evaluate_vision_model(model_lr.to(device), eval_dl)

        params_ratio = params_lr / params_orig
        flops_ratio = flops_lr["total"] / flops_orig["total"]
        delta_flops_ratio = abs(flops_ratio - args.ratio_to_keep)

        print(
            f"[{args.model_name} | iters={k}] loss={eval_lr['loss']:.4f} "
            f"acc={eval_lr['accuracy']:.4f} params_ratio={params_ratio:.4f} "
            f"flops_ratio={flops_ratio:.4f}"
        )
        print(f"delta flops_ratio: {delta_flops_ratio}")

        results.append(
            {
                "model": args.model_name,
                "metric_value": k,
                "loss": float(eval_lr["loss"]),
                "accuracy": float(eval_lr["accuracy"]),
                "params_ratio": float(params_ratio),
                "flops_ratio": float(flops_ratio),
                "try_n_iters": k,
                "delta_flops_ratio": float(delta_flops_ratio),
                "baseline_loss": float(baseline["loss"]),
                "baseline_accuracy": float(baseline["accuracy"]),
                "params_orig": int(params_orig),
                "flops_orig_total": int(flops_orig["total"]),
                "ratio_to_keep": float(args.ratio_to_keep),
            }
        )

    outdir = Path(args.results_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
