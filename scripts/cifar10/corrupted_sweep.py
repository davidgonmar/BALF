# run_cifar10c_auto.py
import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
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
    to_low_rank_activation_aware_auto,  # AUTO methods only
    collect_activation_cache,  # use cached activations
)
from lib.utils.layer_fusion import (
    fuse_batch_norm_inference,
    fuse_conv_bn,
    get_conv_bn_fuse_pairs,
)
from lib.models import load_model


# ---------- CIFAR-10-C Dataset (no TF/TFDS required) ----------
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
    """
    A single corruption at a single severity for CIFAR-10-C.
    Uses numpy memmap to avoid loading all 50k images into RAM.
    """

    def __init__(self, root_dir, corruption, severity, transform):
        assert corruption in CIFAR10C_CORRUPTIONS, f"Unknown corruption: {corruption}"
        assert 1 <= severity <= 5, "severity must be in [1..5]"
        self.transform = transform

        # corruption data: shape (50000, 32, 32, 3), dtype uint8
        cpath = Path(root_dir) / f"{corruption}.npy"
        lpath = Path(root_dir) / "labels.npy"
        if not cpath.exists() or not lpath.exists():
            raise FileNotFoundError(
                f"Missing CIFAR-10-C files. Expected {cpath} and {lpath}."
            )

        self.data = np.load(cpath, mmap_mode="r")
        self.labels = np.load(lpath, mmap_mode="r")

        # Slice indices for the requested severity
        n_per_sev = 10000
        start = (severity - 1) * n_per_sev
        end = severity * n_per_sev
        self.start = start
        self.end = end

        # Basic checks
        if self.data.shape[0] != 50000 or self.labels.shape[0] != 50000:
            raise ValueError(
                "Unexpected CIFAR-10-C shapes; expected 50k images & labels."
            )

    def __len__(self):
        return self.end - self.start

    def __getitem__(self, idx):
        i = self.start + idx
        img = np.array(self.data[i], copy=True)  # move to memory  # (32, 32, 3) uint8
        target = int(self.labels[i])
        img = self.transform(img)

        return img, target


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
    parser.add_argument(
        "--mode",
        default="flops_auto",
        choices=["flops_auto", "params_auto"],
        help="Only AUTO methods are supported in this script.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----- Load model -----
    model = load_model(
        args.model_name,
        pretrained_path=args.pretrained_path,
    ).to(device)

    # ----- Transforms -----
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ]
    )

    # ----- Clean CIFAR-10 (test) -----
    eval_ds = datasets.CIFAR10(
        root="data", train=False, transform=transform, download=True
    )
    eval_dl = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ----- Small calibration subset from train for activations -----
    train_ds = datasets.CIFAR10(
        root="data", train=True, transform=transform, download=True
    )
    # 1024 random samples as in your original script
    subset_idx = torch.randint(0, len(train_ds), (1024,))
    subset = torch.utils.data.Subset(train_ds, subset_idx)
    train_dl = DataLoader(
        subset,
        batch_size=256,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ----- (Optional) BN fusion for stable eval -----
    fuse_pairs = get_conv_bn_fuse_pairs(model)
    model_fused = fuse_conv_bn(
        model, fuse_pairs, fuse_impl=fuse_batch_norm_inference, inplace=False
    ).to(device)

    # ----- Baseline metrics (clean) -----
    baseline_clean = evaluate_vision_model(model_fused, eval_dl)
    params_orig = sum(p.numel() for p in model_fused.parameters())
    flops_orig = count_model_flops(model_fused, (1, 3, 32, 32), formatted=False)

    print(
        f"[baseline/clean] loss={baseline_clean['loss']:.4f} acc={baseline_clean['accuracy']:.4f} "
        f"params={params_orig} flops_total={flops_orig['total']}"
    )

    # ----- Build activation cache once -----
    layer_keys = [k for k in get_all_convs_and_linears(model)]
    activation_cache = collect_activation_cache(model, train_dl, keys=layer_keys)

    # ----- Ratios for AUTO methods -----
    ratios_comp = [0.6, 0.7, 0.8]

    # ----- Prepare output structure -----
    results = {
        "meta": {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "model_name": args.model_name,
            "pretrained_path": str(args.pretrained_path),
            "mode": args.mode,
            "seed": args.seed,
            "cifar10c_root": str(args.cifar10c_root),
            "corruptions": CIFAR10C_CORRUPTIONS,
            "severities": [1, 2, 3, 4, 5],
        },
        "clean": {
            "baseline": {
                "loss": float(baseline_clean["loss"]),
                "accuracy": float(baseline_clean["accuracy"]),
                "params": int(params_orig),
                "flops_total": int(flops_orig["total"]),
            },
            "variants": [],  # filled below
        },
        "cifar10c": [],  # list of {corruption, severity, baseline:{...}, variants:[...]}
    }

    # ----- Evaluate baseline on all CIFAR-10-C subsets -----
    for corr in CIFAR10C_CORRUPTIONS:
        for sev in [1, 2, 3, 4, 5]:
            ds = CIFAR10CSubset(args.cifar10c_root, corr, sev, transform=transform)
            dl = DataLoader(
                ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
            )
            base_metrics = evaluate_vision_model(model_fused, dl)
            results["cifar10c"].append(
                {
                    "corruption": corr,
                    "severity": sev,
                    "baseline": {
                        "loss": float(base_metrics["loss"]),
                        "accuracy": float(base_metrics["accuracy"]),
                    },
                    "variants": [],  # will be filled per variant below
                }
            )
            print(
                f"[baseline/CIFAR10-C {corr} s{sev}] "
                f"loss={base_metrics['loss']:.4f} acc={base_metrics['accuracy']:.4f}"
            )

    # Helper: function to append variant metrics to all matching entries
    def append_variant_result(container_list, corr, sev, variant_entry):
        for rec in container_list:
            if rec["corruption"] == corr and rec["severity"] == sev:
                rec["variants"].append(variant_entry)
                return
        raise RuntimeError(
            "Internal error: missing baseline entry for corruption/severity"
        )

    # ----- Loop over AUTO variants; evaluate clean + all CIFAR-10-C subsets -----
    for k in ratios_comp:
        print(f"\n=== Building AUTO variant: {args.mode} ratio_to_keep={k:.3f} ===")
        model_lr = to_low_rank_activation_aware_auto(
            model,
            activation_cache,  # pass the cache instead of the dataloader
            ratio_to_keep=k,
            inplace=False,
            keys=layer_keys,
            metric="flops" if args.mode == "flops_auto" else "params",
            save_dir=f"./svd-cache/{args.model_name}",
        )
        model_eval = fuse_conv_bn(
            model_lr, fuse_pairs, fuse_impl=fuse_batch_norm_inference, inplace=False
        ).to(device)

        params_lr = sum(p.numel() for p in model_eval.parameters())
        flops_raw_lr = count_model_flops(model_eval, (1, 3, 32, 32), formatted=False)
        flops_ratio = float(flops_raw_lr["total"] / flops_orig["total"])
        params_ratio = float(params_lr / params_orig)

        # Clean eval for this variant
        eval_clean_lr = evaluate_vision_model(model_eval, eval_dl)
        clean_entry = {
            "metric_value": float(k),
            "loss": float(eval_clean_lr["loss"]),
            "accuracy": float(eval_clean_lr["accuracy"]),
            "params_ratio": params_ratio,
            "flops_ratio": flops_ratio,
        }
        results["clean"]["variants"].append(clean_entry)

        print(
            f"[variant/clean ratio={k:.3f}] loss={eval_clean_lr['loss']:.4f} "
            f"acc={eval_clean_lr['accuracy']:.4f} "
            f"params_ratio={params_ratio:.4f} flops_ratio={flops_ratio:.4f}"
        )

        # CIFAR-10-C eval for this variant (iterate corruption then severities)
        for corr in CIFAR10C_CORRUPTIONS:
            # Load each corruption once, iterate severities via slices
            # Use memmap dataset for efficiency
            for sev in [1, 2, 3, 4, 5]:
                ds = CIFAR10CSubset(args.cifar10c_root, corr, sev, transform=transform)
                dl = DataLoader(
                    ds,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=True,
                )
                eval_lr = evaluate_vision_model(model_eval, dl)

                variant_entry = {
                    "metric_value": float(k),
                    "loss": float(eval_lr["loss"]),
                    "accuracy": float(eval_lr["accuracy"]),
                    "params_ratio": params_ratio,
                    "flops_ratio": flops_ratio,
                }
                append_variant_result(results["cifar10c"], corr, sev, variant_entry)

                print(
                    f"[variant/CIFAR10-C {corr} s{sev} ratio={k:.3f}] "
                    f"loss={eval_lr['loss']:.4f} acc={eval_lr['accuracy']:.4f}"
                )

        # free GPU memory between variants
        del model_lr, model_eval
        torch.cuda.empty_cache()

    # ----- Write single JSON file -----
    out_path = Path(args.results_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results JSON to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
