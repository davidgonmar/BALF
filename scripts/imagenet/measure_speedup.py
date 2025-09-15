#!/usr/bin/env python3
import argparse
import time
import statistics
import functools
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import torchvision.models as models
import timm

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from lib.utils import (
    seed_everything,
    imagenet_mean,
    imagenet_std,
    maybe_retrieve_activation_cache,
    make_factorization_cache_location,
    get_all_convs_and_linears,
)
from lib.factorization.factorize import to_low_rank_activation_aware_auto


def build_model(model_name, device):
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
            timm.create_model, model_name="vit_base_patch16_224", pretrained=True
        ),
        "deit_b_16": functools.partial(
            timm.create_model, model_name="deit_base_patch16_224", pretrained=True
        ),
    }
    model = model_dict[model_name]()
    return model.to(device).eval()


def build_calib_loader(model_name, train_dir, calib_size, seed):
    interp_mode = (
        transforms.InterpolationMode.BICUBIC
        if model_name in ["vit_b_16", "deit_b_16"]
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
    ds_full = datasets.ImageFolder(train_dir, transform=tf)
    if calib_size > 0 and calib_size < len(ds_full):
        g = torch.Generator().manual_seed(seed + 12345)
        idx = torch.randperm(len(ds_full), generator=g)[:calib_size]
        ds = Subset(ds_full, idx.tolist())
    else:
        ds = ds_full
    return ds


def benchmark_latencies(model, batch_size, iters, warmup, device):
    torch.backends.cudnn.benchmark = True
    torch.set_grad_enabled(False)
    x = torch.randn(batch_size, 3, 224, 224, device=device)

    for _ in range(warmup):
        with torch.inference_mode():
            _ = model(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            _ = model(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return times


def plot_speedup_single_page(pdf, model_name, ratios, results_by_bs):
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for bs in results_by_bs:
        means = [results_by_bs[bs][r]["speedup_mean"] for r in ratios]
        stds = [results_by_bs[bs][r]["speedup_std"] for r in ratios]
        ax.errorbar(
            ratios, means, yerr=stds, marker="o", linestyle="-", label=f"batch={bs}"
        )
    ax.set_title(f"× Speedup vs. Compression (model={model_name})")
    ax.set_xlabel("Compression ratio kept")
    ax.set_ylabel("× speedup (baseline_latency / compressed_latency)")
    ax.legend(title="Batch size")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument(
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
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calib_size", type=int, default=4096)
    ap.add_argument("--batch_size_cache", type=int, default=128)
    ap.add_argument("--batch_sizes", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model: {args.model_name}")
    model = build_model(args.model_name, device)

    print("Preparing activation cache dataloader ...")
    calib_ds = build_calib_loader(
        args.model_name, args.train_dir, args.calib_size, args.seed
    )
    calib_dl = DataLoader(
        calib_ds, batch_size=args.batch_size_cache, num_workers=8, pin_memory=True
    )

    layer_keys = [k for k in get_all_convs_and_linears(model)]
    activation_cache = maybe_retrieve_activation_cache(
        args.model_name,
        args.calib_size,
        "imagenet",
        "measure_speedup",
        args.seed,
        model,
        calib_dl,
        layer_keys,
    )

    ratios = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # Baseline once per batch size
    base_times = {}
    for bs in args.batch_sizes:
        base_times[bs] = benchmark_latencies(model, bs, args.iters, args.warmup, device)

    results_by_bs = {bs: {} for bs in args.batch_sizes}

    # Stream: compress -> eval -> discard
    for r in ratios:
        print(f"ratio {r:.2f}")
        model_lr = (
            to_low_rank_activation_aware_auto(
                model,
                activation_cache,
                ratio_to_keep=r,
                inplace=False,
                keys=layer_keys,
                metric="flops",
                save_dir=make_factorization_cache_location(
                    args.model_name,
                    args.calib_size,
                    "imagenet",
                    "measure_speedup",
                    args.seed,
                ),
            )
            .to(device)
            .eval()
        )

        for bs in args.batch_sizes:
            comp_times = benchmark_latencies(
                model_lr, bs, args.iters, args.warmup, device
            )
            n = min(len(base_times[bs]), len(comp_times))
            speedups = [base_times[bs][i] / comp_times[i] for i in range(n)]
            mean_s = statistics.mean(speedups)
            std_s = statistics.stdev(speedups) if n > 1 else 0.0
            results_by_bs[bs][r] = {
                "speedup_mean": float(mean_s),
                "speedup_std": float(std_s),
            }

        del model_lr
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{args.model_name}_speedup.pdf"
    with PdfPages(pdf_path) as pdf:
        plot_speedup_single_page(pdf, args.model_name, ratios, results_by_bs)
    print(f"Saved PDF: {pdf_path}")


if __name__ == "__main__":
    main()
