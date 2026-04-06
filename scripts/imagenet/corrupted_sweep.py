import argparse
import functools
import json
from pathlib import Path

import timm
import torch
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from lib.factorization.factorize import (
    collect_activation_cache,
    to_low_rank_activation_aware_auto,
)
from lib.utils import (
    count_model_flops,
    evaluate_vision_model,
    get_all_convs_and_linears,
    imagenet_mean,
    imagenet_std,
    make_factorization_cache_location,
    seed_everything,
)

IMAGENETC_CORRUPTIONS = [
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
    "speckle_noise",
    "gaussian_blur",
    "spatter",
    "saturate",
]

IMAGENETC_SUBSETS = {
    "noise": [
        "gaussian_noise",
        "shot_noise",
        "impulse_noise",
    ],
    "blur": [
        "defocus_blur",
        "glass_blur",
        "motion_blur",
        "zoom_blur",
    ],
    "weather": [
        "snow",
        "frost",
        "fog",
        "brightness",
    ],
    "digital": [
        "contrast",
        "elastic_transform",
        "pixelate",
        "jpeg_compression",
    ],
    "extra": [
        "speckle_noise",
        "gaussian_blur",
        "spatter",
        "saturate",
    ],
}


def build_imagenet_model(model_name):
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
            models.resnext50_32x4d,
            weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1,
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
    return model_dict[model_name]()


def build_imagenet_transform(model_name):
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

    return transforms.Compose(
        [
            transforms.Resize(resize, interpolation=interp_mode),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=ds_mean, std=ds_std),
        ]
    )


class ImageNetCSeveritySubset(Dataset):
    def __init__(
        self, imagenetc_root, imagenetc_subset, corruption, severity, transform
    ):
        assert corruption in IMAGENETC_CORRUPTIONS, f"Unknown corruption: {corruption}"
        assert 1 <= severity <= 5, "severity must be in [1..5]"

        self.transform = transform
        self.corruption_dir = self._resolve_corruption_dir(
            Path(imagenetc_root), imagenetc_subset, corruption
        )
        self.severity_dir = self.corruption_dir / str(severity)
        if not self.severity_dir.exists():
            raise FileNotFoundError(
                f"Missing severity directory for corruption '{corruption}': {self.severity_dir}"
            )

        self.ds = datasets.ImageFolder(str(self.severity_dir), transform=self.transform)

    @staticmethod
    def _resolve_corruption_dir(root, imagenetc_subset, corruption):
        subset_root = root / imagenetc_subset
        if not subset_root.is_dir():
            raise FileNotFoundError(
                f"Missing ImageNet-C subset directory: {subset_root}. "
                "Expected one of noise/, blur/, weather/, digital/, or extra/ under the provided root."
            )

        direct = subset_root / corruption
        if direct.is_dir():
            return direct

        matches = [p for p in subset_root.glob(f"**/{corruption}") if p.is_dir()]
        if not matches:
            raise FileNotFoundError(
                f"Could not find corruption folder '{corruption}' under subset {subset_root}."
            )

        matches = sorted({p.resolve() for p in matches})
        if len(matches) > 1:
            raise RuntimeError(
                f"Found multiple candidate folders for corruption '{corruption}': {matches}. "
                "Please keep only one extracted copy under the provided ImageNet-C root."
            )
        return matches[0]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.ds[idx]


def build_clean_calibration_loader(train_ds, calib_size, batch_size, num_workers):
    subset_idx = torch.randperm(len(train_ds))[:calib_size].tolist()
    subset = Subset(train_ds, subset_idx)
    return DataLoader(
        subset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )


def build_imagenetc_calibration_loader(
    imagenetc_root,
    imagenetc_subset,
    corruption,
    severity,
    calib_size,
    transform,
    batch_size,
    num_workers,
):
    ds_full = ImageNetCSeveritySubset(
        imagenetc_root=imagenetc_root,
        imagenetc_subset=imagenetc_subset,
        corruption=corruption,
        severity=severity,
        transform=transform,
    )
    subset_idx = torch.randperm(len(ds_full))[:calib_size].tolist()
    subset = Subset(ds_full, subset_idx)

    return DataLoader(
        subset,
        batch_size=batch_size,
        num_workers=num_workers,
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
            f"imagenet_{cache_tag}",
            "corrupted_sweep",
            args.seed,
        ),
    )
    model_lr.eval()

    params_lr = sum(p.numel() for p in model_lr.parameters())
    flops_lr = count_model_flops(model_lr, (1, 3, 224, 224))
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
    parser.add_argument(
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
    parser.add_argument(
        "--results_json",
        required=True,
        help="Path to write a single results JSON file.",
    )
    parser.add_argument(
        "--imagenetc_root",
        required=True,
        help=(
            "Root folder containing extracted ImageNet-C download subfolders, e.g. "
            "noise/, blur/, weather/, digital/, extra/."
        ),
    )
    parser.add_argument(
        "--imagenetc_subset",
        required=True,
        choices=sorted(IMAGENETC_SUBSETS.keys()),
        help="Specific ImageNet-C download subset to run: noise, blur, weather, digital, or extra.",
    )
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--val_dir", required=True)
    parser.add_argument(
        "--mode",
        default="flops_auto",
        choices=["flops_auto", "params_auto"],
        help="Only AUTO methods are supported in this script.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size_eval", type=int, default=512)
    parser.add_argument("--batch_size_cache", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--calib_size", type=int, default=8192)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = build_imagenet_transform(args.model_name)

    model = build_imagenet_model(args.model_name).to(device)
    model.eval()

    eval_ds = datasets.ImageFolder(args.val_dir, transform=transform)

    eval_dl = DataLoader(
        eval_ds,
        batch_size=args.batch_size_eval,
        num_workers=args.num_workers,
        shuffle=False,
    )

    train_ds = datasets.ImageFolder(args.train_dir, transform=transform)

    baseline_clean = evaluate_vision_model(model, eval_dl)
    params_orig = sum(p.numel() for p in model.parameters())
    flops_orig = count_model_flops(model, (1, 3, 224, 224))
    flops_orig_total = int(flops_orig["total"])

    print(
        f"[baseline/clean-val] loss={baseline_clean['loss']:.4f} "
        f"acc={baseline_clean['accuracy']:.4f} "
        f"params={params_orig} flops_total={flops_orig_total}"
    )

    layer_keys = get_all_convs_and_linears(model)
    ratios_comp = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    subset_corruptions = IMAGENETC_SUBSETS[args.imagenetc_subset]

    results = {
        "imagenetc_subset": args.imagenetc_subset,
        "clean_test": {
            "baseline": {
                "loss": float(baseline_clean["loss"]),
                "accuracy": float(baseline_clean["accuracy"]),
                "params": int(params_orig),
                "flops_total": flops_orig_total,
            },
            "clean_calibration_variants": [],
            "corrupted_calibration_variants": [],
        },
    }

    clean_calib_dl = build_clean_calibration_loader(
        train_ds=train_ds,
        calib_size=args.calib_size,
        batch_size=args.batch_size_cache,
        num_workers=args.num_workers,
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
            f"[clean calibration -> clean val ratio={k:.6f}] "
            f"loss={rec['loss']:.4f} acc={rec['accuracy']:.4f} "
            f"params_ratio={rec['params_ratio']:.4f} "
            f"flops_ratio={rec['flops_ratio']:.4f}"
        )

    del clean_activation_cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for corr in subset_corruptions:
        for sev in [1, 2, 3, 4, 5]:
            calib_dl = build_imagenetc_calibration_loader(
                imagenetc_root=args.imagenetc_root,
                imagenetc_subset=args.imagenetc_subset,
                corruption=corr,
                severity=sev,
                calib_size=args.calib_size,
                transform=transform,
                batch_size=args.batch_size_cache,
                num_workers=args.num_workers,
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
                    f"[{corr} s{sev} calibration -> clean val ratio={k:.6f}] "
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
