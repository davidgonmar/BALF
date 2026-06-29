import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
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


CIFAR10C_CORRUPTIONS = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
]


class CIFAR10CSubset(Dataset):
    def __init__(self, root_dir, corruption, severity, transform):
        assert corruption in CIFAR10C_CORRUPTIONS, f"Unknown corruption: {corruption}"
        assert 1 <= severity <= 5, "severity must be in [1..5]"

        self.transform = transform

        cpath = Path(root_dir) / f"{corruption}.npy"
        lpath = Path(root_dir) / "labels.npy"
        if not cpath.exists() or not lpath.exists():
            raise FileNotFoundError(
                f"Missing CIFAR-10-C files. Expected {cpath} and {lpath}."
            )

        self.data = np.load(cpath, mmap_mode="r")
        self.labels = np.load(lpath, mmap_mode="r")

        if self.data.shape[0] != 50000 or self.labels.shape[0] != 50000:
            raise ValueError(
                "Unexpected CIFAR-10-C shapes; expected 50k images & labels."
            )

        n_per_sev = 10000
        self.start = (severity - 1) * n_per_sev
        self.end = severity * n_per_sev

    def __len__(self):
        return self.end - self.start

    def __getitem__(self, idx):
        i = self.start + idx
        img = np.array(self.data[i], copy=True)
        target = int(self.labels[i])
        img = self.transform(img)
        return img, target


def build_clean_calibration_loader(train_ds, calib_size, num_workers, seed):
    g = torch.Generator().manual_seed(seed + 12345)
    subset_idx = torch.randperm(len(train_ds), generator=g)[:calib_size].tolist()
    subset = Subset(train_ds, subset_idx)
    return DataLoader(
        subset,
        batch_size=256,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
    )


def build_cifar10c_calibration_loader(
    cifar10c_root, corruption, severity, calib_size, transform, num_workers, seed
):
    ds_full = CIFAR10CSubset(
        root_dir=cifar10c_root,
        corruption=corruption,
        severity=severity,
        transform=transform,
    )

    g = torch.Generator().manual_seed(seed + 12345)
    subset_idx = torch.randperm(len(ds_full), generator=g)[:calib_size].tolist()
    subset = Subset(ds_full, subset_idx)

    return DataLoader(
        subset,
        batch_size=256,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
    )


def compress_with_cache_and_eval_clean(
    model,
    activation_cache,
    layer_keys,
    ratio_to_keep,
    args,
    eval_dl,
    params_orig,
    flops_orig_total,
    cache_tag,
):
    model_lr = to_low_rank_activation_aware_auto(
        model,
        activation_cache,
        ratio_to_keep=ratio_to_keep,
        inplace=False,
        keys=layer_keys,
        metric="flops" if args.mode == "flops_auto" else "params",
        save_dir=make_factorization_cache_location(
            args.model_name,
            args.calib_size,
            f"cifar10_{cache_tag}",
            "corrupted_sweep",
            args.seed,
        ),
    )
    model_lr.eval()

    params_lr = sum(p.numel() for p in model_lr.parameters())
    flops_lr = count_model_flops(model_lr, (1, 3, 32, 32))

    eval_clean = evaluate_vision_model(model_lr, eval_dl)

    result = {
        "metric_value": float(ratio_to_keep),
        "loss": float(eval_clean["loss"]),
        "accuracy": float(eval_clean["accuracy"]),
        "params_ratio": float(params_lr / params_orig),
        "flops_ratio": float(flops_lr["total"] / flops_orig_total),
    }

    del model_lr
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model_name", required=True, choices=["resnet20", "resnet56"])
    parser.add_argument("--pretrained_path", required=True)
    parser.add_argument(
        "--results_json",
        required=True,
        help="Path to write a single results JSON file.",
    )
    parser.add_argument(
        "--cifar10c_root",
        required=True,
        help="Directory containing CIFAR-10-C .npy files.",
    )
    parser.add_argument("--data_root", default="data")
    parser.add_argument(
        "--mode",
        default="flops_auto",
        choices=["flops_auto", "params_auto"],
        help="Only AUTO methods are supported in this script.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--calib_size", type=int, default=1024)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ]
    )

    model = load_model(
        args.model_name,
        pretrained_path=args.pretrained_path,
    ).to(device)
    model.eval()

    eval_ds = datasets.CIFAR10(
        root=args.data_root, train=False, transform=transform, download=True
    )
    eval_dl = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
    )

    train_ds = datasets.CIFAR10(
        root=args.data_root, train=True, transform=transform, download=True
    )

    baseline_clean = evaluate_vision_model(model, eval_dl)
    params_orig = sum(p.numel() for p in model.parameters())
    flops_orig = count_model_flops(model, (1, 3, 32, 32))
    flops_orig_total = int(flops_orig["total"])

    print(
        f"[baseline/clean-test] loss={baseline_clean['loss']:.4f} "
        f"acc={baseline_clean['accuracy']:.4f} "
        f"params={params_orig} flops_total={flops_orig_total}"
    )

    layer_keys = get_all_convs_and_linears(model)
    ratios_comp = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    results = {
        "clean_test": {
            "baseline": {
                "loss": float(baseline_clean["loss"]),
                "accuracy": float(baseline_clean["accuracy"]),
                "params": int(params_orig),
                "flops_total": flops_orig_total,
            },
            "clean_calibration_variants": [],
            "corrupted_calibration_variants": [],
        }
    }

    # Clean CIFAR-10 calibration
    clean_calib_dl = build_clean_calibration_loader(
        train_ds=train_ds,
        calib_size=args.calib_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    model.eval()
    clean_activation_cache = collect_activation_cache(
        model, clean_calib_dl, keys=layer_keys
    )

    for k in ratios_comp:
        rec = compress_with_cache_and_eval_clean(
            model=model,
            activation_cache=clean_activation_cache,
            layer_keys=layer_keys,
            ratio_to_keep=k,
            args=args,
            eval_dl=eval_dl,
            params_orig=params_orig,
            flops_orig_total=flops_orig_total,
            cache_tag="clean",
        )
        results["clean_test"]["clean_calibration_variants"].append(rec)

        print(
            f"[clean calibration -> clean test ratio={k:.3f}] "
            f"loss={rec['loss']:.4f} acc={rec['accuracy']:.4f} "
            f"params_ratio={rec['params_ratio']:.4f} "
            f"flops_ratio={rec['flops_ratio']:.4f}"
        )

    del clean_activation_cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Corrupted CIFAR-10-C calibration
    for corr in CIFAR10C_CORRUPTIONS:
        for sev in [1, 2, 3, 4, 5]:
            calib_dl = build_cifar10c_calibration_loader(
                cifar10c_root=args.cifar10c_root,
                corruption=corr,
                severity=sev,
                calib_size=args.calib_size,
                transform=transform,
                num_workers=args.num_workers,
                seed=args.seed,
            )

            model.eval()
            activation_cache = collect_activation_cache(
                model, calib_dl, keys=layer_keys
            )

            entry = {
                "corruption": corr,
                "severity": sev,
                "variants": [],
            }

            for k in ratios_comp:
                rec = compress_with_cache_and_eval_clean(
                    model=model,
                    activation_cache=activation_cache,
                    layer_keys=layer_keys,
                    ratio_to_keep=k,
                    args=args,
                    eval_dl=eval_dl,
                    params_orig=params_orig,
                    flops_orig_total=flops_orig_total,
                    cache_tag=f"corrupted_{corr}_s{sev}",
                )
                entry["variants"].append(rec)

                print(
                    f"[{corr} s{sev} calibration -> clean test ratio={k:.3f}] "
                    f"loss={rec['loss']:.4f} acc={rec['accuracy']:.4f} "
                    f"params_ratio={rec['params_ratio']:.4f} "
                    f"flops_ratio={rec['flops_ratio']:.4f}"
                )

            results["clean_test"]["corrupted_calibration_variants"].append(entry)

            del activation_cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    out_path = Path(args.results_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results JSON to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
