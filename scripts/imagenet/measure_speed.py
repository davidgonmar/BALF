import argparse
import json
import time
import statistics
import functools
from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import torchvision.models as models
import timm

import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "scripts"))

from plot_style import (
    FIGSIZE_WIDE,
    RATIO_COLORS,
    apply_paper_style,
    save_pdf,
    style_axes,
)

from lib.utils import (
    seed_everything,
    imagenet_mean,
    imagenet_std,
    maybe_retrieve_activation_cache,
    make_factorization_cache_location,
    get_all_convs_and_linears,
    count_model_flops,
)
from lib.factorization.factorize import to_low_rank_activation_aware_auto

apply_paper_style()


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
    model = model_dict[model_name]()
    return model.to(device).eval()


def build_calib_loader(model_name, train_dir, calib_size, seed):
    interp_mode = (
        transforms.InterpolationMode.BICUBIC
        if model_name in ["vit_b_16", "deit_b_16"]
        else transforms.InterpolationMode.BILINEAR
    )
    ds_mean, ds_std = (
        ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        if model_name == "vit_b_16"
        else (imagenet_mean, imagenet_std)
    )
    resize = 248 if model_name in ["vit_b_16", "deit_b_16"] else 256
    tf = transforms.Compose(
        [
            transforms.Resize(resize, interpolation=interp_mode),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=ds_mean, std=ds_std),
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


@torch.inference_mode()
def timing_single_measure(model, batch_size, throughput_batches, warmup, device):
    model.to(device).eval()
    torch.backends.cudnn.benchmark = True
    try:
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
        elapsed = t1 - t0
        items = throughput_batches * batch_size
        return {
            "throughput": items / elapsed,
            "latency_ms": 1000.0 * elapsed / throughput_batches,
        }
    except RuntimeError as err:
        if "out of memory" not in str(err).lower():
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"batch {batch_size}: CUDA out of memory, skipping")
        return None


def measure_timing_repeats(
    model, batch_sizes, throughput_batches, warmup, repeats, device
):
    out = {bs: [] for bs in batch_sizes}
    for bs in batch_sizes:
        for _ in range(repeats):
            timing = timing_single_measure(
                model, bs, throughput_batches, warmup, device
            )
            if timing is None:
                out[bs] = None
                break
            out[bs].append(timing)
    return out


def mean_std(vals):
    if not vals:
        return None, None
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return float(m), float(s)


def timing_mean_std(vals):
    if not vals:
        return None, None, None, None
    throughput_mean, throughput_std = mean_std([v["throughput"] for v in vals])
    latency_mean, latency_std = mean_std([v["latency_ms"] for v in vals])
    return throughput_mean, throughput_std, latency_mean, latency_std


def stat_entry(vals, target_ratio, flops_ratio):
    if not vals:
        return {
            "target_ratio": float(target_ratio),
            "flops_ratio": float(flops_ratio),
            "throughput_mean": None,
            "throughput_std": None,
            "throughput_runs": None,
            "latency_ms_mean": None,
            "latency_ms_std": None,
            "latency_ms_runs": None,
        }
    throughputs = [float(v["throughput"]) for v in vals]
    latencies = [float(v["latency_ms"]) for v in vals]
    thpt_m, thpt_s = mean_std(throughputs)
    lat_m, lat_s = mean_std(latencies)
    return {
        "target_ratio": float(target_ratio),
        "flops_ratio": float(flops_ratio),
        "throughput_mean": float(thpt_m),
        "throughput_std": float(thpt_s),
        "throughput_runs": throughputs,
        "latency_ms_mean": float(lat_m),
        "latency_ms_std": float(lat_s),
        "latency_ms_runs": latencies,
    }


def plot_metric(pdf_path, model_name, gpu_label, results_by_bs, metric, ylabel):
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    for i, bs in enumerate(sorted(results_by_bs)):
        entries = [
            entry
            for entry in results_by_bs[bs].values()
            if entry[f"{metric}_mean"] is not None
        ]
        entries.sort(key=lambda entry: entry["flops_ratio"])
        xs = [entry["flops_ratio"] for entry in entries]
        means = [entry[f"{metric}_mean"] for entry in entries]
        stds = [entry[f"{metric}_std"] for entry in entries]
        if not xs:
            continue
        ax.errorbar(
            xs,
            means,
            yerr=stds,
            color=RATIO_COLORS[i % len(RATIO_COLORS)],
            marker="o",
            linestyle="-",
            markerfacecolor="none",
            markeredgewidth=0.8,
            linewidth=1.25,
            markersize=3.0,
            elinewidth=0.7,
            capsize=2.0,
            capthick=0.7,
            label=f"Batch {bs}",
        )
    ax.set_title(f"{PRETTY_MODEL_NAMES.get(model_name, model_name)} ({gpu_label})", pad=3)
    ax.set_xlabel("FLOPs ratio")
    ax.set_ylabel(ylabel)
    style_axes(ax)
    ax.legend(
        title="Batch size",
        loc="lower left",
        bbox_to_anchor=(0.02, 0.02),
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.85,
        ncol=1,
        handlelength=1.8,
        handletextpad=0.45,
    )
    fig.tight_layout(pad=0.25)
    save_pdf(fig, pdf_path)
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
    ap.add_argument("--gpu_label", default=None)
    ap.add_argument("--output_tag", required=True)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_label = args.gpu_label or (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    )

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
    flops_orig = count_model_flops(model, (1, 3, 224, 224))["total"]
    ratios = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]

    base_timing_runs = measure_timing_repeats(
        model,
        args.batch_sizes,
        args.throughput_batches,
        args.throughput_warmup,
        args.throughput_repeats,
        device,
    )
    for bs in args.batch_sizes:
        thpt_m, thpt_s, lat_m, lat_s = timing_mean_std(base_timing_runs[bs])
        if thpt_m is None:
            print(f"baseline batch {bs}: skipped")
        else:
            print(
                f"baseline batch {bs}: "
                f"{thpt_m:.2f} images/s (+/- {thpt_s:.2f}), "
                f"{lat_m:.3f} ms/batch (+/- {lat_s:.3f})"
            )

    results_by_bs = {bs: {} for bs in args.batch_sizes}

    for bs in args.batch_sizes:
        results_by_bs[bs][1.0] = stat_entry(base_timing_runs[bs], 1.0, 1.0)

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
        flops_lr = count_model_flops(model_lr, (1, 3, 224, 224))["total"]
        flops_ratio = float(flops_lr / flops_orig)
        print(f"  actual flops ratio: {flops_ratio:.4f}")

        comp_timing_runs = measure_timing_repeats(
            model_lr,
            args.batch_sizes,
            args.throughput_batches,
            args.throughput_warmup,
            args.throughput_repeats,
            device,
        )

        for bs in args.batch_sizes:
            thpt_m, thpt_s, lat_m, lat_s = timing_mean_std(comp_timing_runs[bs])
            results_by_bs[bs][r] = stat_entry(comp_timing_runs[bs], r, flops_ratio)
            if thpt_m is None:
                print(f"  batch {bs}: skipped")
            else:
                print(
                    f"  batch {bs}: "
                    f"{thpt_m:.2f} images/s (+/- {thpt_s:.2f}), "
                    f"{lat_m:.3f} ms/batch (+/- {lat_s:.3f})"
                )

        del model_lr
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    target_ratios = [1.0] + ratios
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{args.model_name}_speed_{args.output_tag}.json"
    throughput_pdf_path = out_dir / f"{args.model_name}_throughput_{args.output_tag}.pdf"
    latency_pdf_path = out_dir / f"{args.model_name}_latency_{args.output_tag}.pdf"
    with open(json_path, "w") as f:
        json.dump(
            {
                "model_name": args.model_name,
                "gpu_label": gpu_label,
                "output_tag": args.output_tag,
                "seed": args.seed,
                "calib_size": args.calib_size,
                "batch_size_cache": args.batch_size_cache,
                "batch_sizes": args.batch_sizes,
                "throughput_batches": args.throughput_batches,
                "throughput_warmup": args.throughput_warmup,
                "throughput_repeats": args.throughput_repeats,
                "target_ratios": target_ratios,
                "results_by_batch_size": {
                    str(bs): {str(r): entry for r, entry in by_ratio.items()}
                    for bs, by_ratio in results_by_bs.items()
                },
            },
            f,
            indent=2,
        )
    plot_metric(
        throughput_pdf_path,
        args.model_name,
        gpu_label,
        results_by_bs,
        "throughput",
        "Throughput (images/s)",
    )
    plot_metric(
        latency_pdf_path,
        args.model_name,
        gpu_label,
        results_by_bs,
        "latency_ms",
        "Latency (ms/batch)",
    )
    print(f"Saved PDF: {throughput_pdf_path}")
    print(f"Saved PDF: {latency_pdf_path}")
    print(f"Saved JSON: {json_path}")


if __name__ == "__main__":
    main()
