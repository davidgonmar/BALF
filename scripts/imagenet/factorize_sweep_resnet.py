import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from lib.utils import (
    evaluate_vision_model,
    seed_everything,
    count_model_flops,
    get_all_convs_and_linears,
)
from lib.factorization.factorize import (
    to_low_rank_activation_aware_auto,
    to_low_rank_activation_aware_manual,
    to_low_rank_manual,
    collect_activation_cache,
)
from lib.utils.layer_fusion import (
    fuse_batch_norm_inference,
    fuse_conv_bn,
    get_conv_bn_fuse_pairs,
)
import torchvision.models as models

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--model_name",
    required=True,
    choices=["resnet18", "resnet34", "resnet50", "mobilenet_v2"],
)
parser.add_argument("--results_dir", required=True)
parser.add_argument(
    "--mode",
    default="flops_auto",
    choices=["flops_auto", "params_auto", "energy_act_aware", "energy"],
)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--train_dir", required=True)
parser.add_argument("--val_dir", required=True)
parser.add_argument("--batch_size_eval", type=int, default=256)
parser.add_argument("--batch_size_cache", type=int, default=128)
parser.add_argument("--subset_size", type=int, default=4096)
parser.add_argument("--cache_file", type=str, default="activation_cache.pt")
parser.add_argument("--force_recache", action="store_true")
args = parser.parse_args()
# args.force_recache = True
seed_everything(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_dict = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "mobilenet_v2": models.mobilenet_v2,
}
model = model_dict[args.model_name](pretrained=True).to(device)


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
train_tf = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ]
)

eval_ds = datasets.ImageFolder(args.val_dir, transform=eval_tf)
eval_dl = DataLoader(
    eval_ds,
    batch_size=args.batch_size_eval,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
)
# subset eval_dl to 1000 images
eval_ds = Subset(eval_ds, torch.randperm(len(eval_ds))[:3000])
eval_dl = DataLoader(
    eval_ds,
    batch_size=args.batch_size_eval,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
)


train_ds_full = datasets.ImageFolder(args.train_dir, transform=train_tf)

if args.subset_size > 0 and args.subset_size < len(train_ds_full):
    idx = torch.randint(0, len(train_ds_full), (args.subset_size,))
    train_ds = Subset(train_ds_full, idx.tolist())
else:
    train_ds = train_ds_full
train_dl = DataLoader(
    train_ds,
    batch_size=args.batch_size_cache,
    shuffle=True,
    drop_last=True,
    num_workers=8,
    pin_memory=True,
)

fuse_pairs = get_conv_bn_fuse_pairs(model)
model_fused = fuse_conv_bn(
    model, fuse_pairs, fuse_impl=fuse_batch_norm_inference, inplace=False
)

baseline_metrics = evaluate_vision_model(model_fused, eval_dl)
params_orig = sum(p.numel() for p in model_fused.parameters())
flops_orig = count_model_flops(model_fused, (1, 3, 224, 224), formatted=False)

print(
    f"[original] loss={baseline_metrics['loss']:.4f} acc={baseline_metrics['accuracy']:.4f} "
    f"params={params_orig} flops_total={flops_orig['total']}"
)

"""
    """
ratios_comp = [
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.45,
    0.5,
    0.55,
    0.6,
    0.65,
    0.7,
    0.75,
    0.8,
    0.85,
    0.9,
    0.95,
    1.00,
]

ratios_energy = [
    0.01,
    0.05,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    0.95,
    0.99,
    0.992,
    0.995,
    0.997,
    0.999,
    0.9999,
    0.99999,
    0.999999,
]

fuse_conv_bn = lambda *args, **kwargs: args[0]
layer_keys = [k for k in get_all_convs_and_linears(model)]
# remove linear
layer_keys = [k for k in layer_keys if "linear" not in k and "fc" not in k]
# print(layer_keys)
base_dir = Path(args.results_dir)
base_dir.mkdir(parents=True, exist_ok=True)
cache_path = base_dir / args.cache_file

if cache_path.exists() and not args.force_recache:
    activation_cache = torch.load(cache_path, map_location="cpu")
else:
    activation_cache = collect_activation_cache(model, train_dl, keys=layer_keys)
    torch.save(activation_cache, cache_path)

results = []

for k in (
    ratios_comp
    if args.mode == "flops_auto" or args.mode == "params_auto"
    else ratios_energy
):
    if args.mode == "flops_auto" or args.mode == "params_auto":
        model_lr = to_low_rank_activation_aware_auto(
            model,
            activation_cache,
            ratio_to_keep=k,
            inplace=False,
            keys=layer_keys,
            metric="flops" if args.mode == "flops_auto" else "params",
            save_dir="./svd-cache/" + args.model_name,
        )
    elif args.mode == "energy_act_aware":
        model_lr = to_low_rank_activation_aware_manual(
            model,
            activation_cache,
            cfg_dict={
                kk: {"name": "svals_energy_ratio_to_keep", "value": k}
                for kk in layer_keys
            },
            inplace=False,
            save_dir="./svd-cache/" + args.model_name,
        )
    elif args.mode == "energy":
        model_lr = to_low_rank_manual(
            model,
            cfg_dict={
                kk: {"name": "svals_energy_ratio_to_keep", "value": k}
                for kk in layer_keys
            },
            inplace=False,
        )
    # print(model_lr)
    model_eval = fuse_conv_bn(
        model_lr, fuse_pairs, fuse_impl=fuse_batch_norm_inference, inplace=False
    )
    # print(model_eval)
    # print(torch.cuda.memory_allocated() / 1e6, "MB allocated after model eval")
    params_lr = sum(p.numel() for p in model_eval.parameters())
    # print(params_lr / params_orig)
    flops_raw_lr = count_model_flops(model_eval, (1, 3, 224, 224), formatted=False)
    eval_lr = evaluate_vision_model(model_eval.to(device), eval_dl)

    print(
        f"[ratio={k:.6f}] loss={eval_lr['loss']:.4f} acc={eval_lr['accuracy']:.4f} "
        f"params_ratio={params_lr/params_orig:.4f} flops_ratio={flops_raw_lr['total']/flops_orig['total']:.4f}"
    )

    ratio_dir = base_dir / args.mode
    ratio_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = ratio_dir / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump(
            {
                "metric_value": k,
                "loss": float(eval_lr["loss"]),
                "accuracy": float(eval_lr["accuracy"]),
                "params_ratio": float(params_lr / params_orig),
                "flops_ratio": float(flops_raw_lr["total"] / flops_orig["total"]),
                "mode": args.mode,
                "model_name": args.model_name,
                "seed": args.seed,
            },
            f,
            indent=2,
        )

    results.append(
        {
            "metric_value": k,
            "loss": float(eval_lr["loss"]),
            "accuracy": float(eval_lr["accuracy"]),
            "params_ratio": float(params_lr / params_orig),
            "flops_ratio": float(flops_raw_lr["total"] / flops_orig["total"]),
            "mode": args.mode,
        }
    )

output_file = base_dir / "results.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)
