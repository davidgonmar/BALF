import argparse
import time
import statistics
import functools
from pathlib import Path

import torch
import torch.nn as nn
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


def replace_bn_with_identity(module):
    for name, child in list(module.named_children()):
        if isinstance(
            child, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
        ):
            setattr(module, name, nn.Identity())
        else:
            replace_bn_with_identity(child)


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


PRETTY_MODEL_NAMES = {
    "resnet18": "ResNet-18",
    "resnet50": "ResNet-50",
    "mobilenet_v2": "MobileNetV2",
    "resnext50_32x4d": r"ResNeXt-50 (32$\times$4d)",
    "resnext101_32x8d": r"ResNeXt-101 (32$\times$8d)",
    "vit_b_16": "ViT-B/16",
    "deit_b_16": "DeiT-B/16",
}


@torch.no_grad()
def throughput_single_measure(model, batch_size, throughput_batches, warmup, device):
    model.to(device).eval()
    torch.backends.cudnn.benchmark = True
    x = torch.randn(batch_size, 3, 224, 224, device=device)
    for _ in range(warmup):
        _ = model(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(throughput_batches):
        _ = model(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    items = throughput_batches * batch_size
    return items / (t1 - t0)


def measure_throughput_repeats(
    model, batch_sizes, throughput_batches, warmup, repeats, device
):
    out = {bs: [] for bs in batch_sizes}
    for bs in batch_sizes:
        for _ in range(repeats):
            thpt = throughput_single_measure(
                model, bs, throughput_batches, warmup, device
            )
            out[bs].append(float(thpt))
    return out


def mean_std(vals):
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return float(m), float(s)


def plot_throughput(pdf, model_name, ratios, results_by_bs):
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for bs in results_by_bs:
        means = [results_by_bs[bs][r]["thpt_mean"] for r in ratios]
        stds = [results_by_bs[bs][r]["thpt_std"] for r in ratios]
        ax.errorbar(
            ratios, means, yerr=stds, marker="o", linestyle="-", label=f"batch={bs}"
        )
    ax.set_title(PRETTY_MODEL_NAMES.get(model_name, model_name))
    ax.set_xlabel("FLOPs ratio retained")
    ax.set_ylabel("Throughput (items/sec)")
    ax.legend(title="Batch size")
    ax.grid(True)
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
    ap.add_argument("--calib_size", type=int, default=8192)
    ap.add_argument("--batch_size_cache", type=int, default=128)
    ap.add_argument("--batch_sizes", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--throughput_batches", type=int, default=30)
    ap.add_argument("--throughput_warmup", type=int, default=5)
    ap.add_argument("--throughput_repeats", type=int, default=5)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(args.model_name, device)

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
        "measure_speed",
        args.seed,
        model,
        calib_dl,
        layer_keys,
    )

    replace_bn_with_identity(model)
    ratios = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]

    base_throughput_runs = measure_throughput_repeats(
        model,
        args.batch_sizes,
        args.throughput_batches,
        args.throughput_warmup,
        args.throughput_repeats,
        device,
    )
    base_throughput_stats = {
        bs: mean_std(base_throughput_runs[bs]) for bs in args.batch_sizes
    }
    for bs in args.batch_sizes:
        m, s = base_throughput_stats[bs]
        print(f"baseline throughput batch {bs}: {m:.2f} it/s (+/- {s:.2f})")

    results_by_bs = {bs: {} for bs in args.batch_sizes}

    for bs in args.batch_sizes:
        m, s = base_throughput_stats[bs]
        results_by_bs[bs][1.0] = {
            "thpt_mean": float(m),
            "thpt_std": float(s),
        }

    for r in [x for x in ratios if x != 1.0]:
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
                    "measure_speed",
                    args.seed,
                ),
            )
            .to(device)
            .eval()
        )

        comp_throughput_runs = measure_throughput_repeats(
            model_lr,
            args.batch_sizes,
            args.throughput_batches,
            args.throughput_warmup,
            args.throughput_repeats,
            device,
        )

        for bs in args.batch_sizes:
            comp_mean, comp_std = mean_std(comp_throughput_runs[bs])
            results_by_bs[bs][r] = {
                "thpt_mean": float(comp_mean),
                "thpt_std": float(comp_std),
            }
            print(f"  batch {bs}: {comp_mean:.2f} it/s (+/- {comp_std:.2f})")

        del model_lr
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # add 1.0 to ratios for plotting
    ratios = [1.0] + ratios
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{args.model_name}_throughput.pdf"
    with PdfPages(pdf_path) as pdf:
        plot_throughput(pdf, args.model_name, ratios, results_by_bs)
    print(f"Saved PDF: {pdf_path}")


if __name__ == "__main__":
    main()
