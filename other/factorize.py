import argparse
import json
import re
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from utils import (
    cifar10_mean,
    cifar10_std,
    evaluate_vision_model,
    load_vision_model,
    seed_everything,
    count_model_flops,
)
from factorization.utils import (
    get_all_convs_and_linears,
)
from factorization.factorize import to_low_rank_activation_aware_global

from utils.layer_fusion import (
    fuse_batch_norm_inference,
    fuse_conv_bn,
    resnet20_fuse_pairs,
)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--model_name", default="resnet20")
parser.add_argument(
    "--pretrained_path",
    default="./resnet20.pth",
)
parser.add_argument(
    "--output_file", required=True, help="JSON file to collect per‑ratio metrics."
)
parser.add_argument(
    "--metric",
    default="params",
    choices=["flops", "params"],
    help="What the low‑rank search optimises.",
)
parser.add_argument(
    "--models_dir",
    default="compressed_models",
    help="Directory into which *.pt compressed models are written.",
)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
seed_everything(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

m = re.search(r"reg([0-9]*\.?[0-9]+)", args.pretrained_path)
reg_strength: float | None = float(m.group(1)) if m else None

model = load_vision_model(
    args.model_name,
    pretrained_path=args.pretrained_path,
    strict=True,
    model_args={"num_classes": 10},
).to(device)

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(cifar10_mean, cifar10_std)]
)

test_ds = datasets.CIFAR10(root="data", train=False, transform=transform, download=True)
test_dl = DataLoader(test_ds, batch_size=512, shuffle=False)

train_ds = datasets.CIFAR10(root="data", train=True, transform=transform, download=True)
subset = torch.utils.data.Subset(train_ds, torch.randint(0, len(train_ds), (1024,)))
train_dl = DataLoader(subset, batch_size=256, shuffle=True, drop_last=True)

model_fused = fuse_conv_bn(
    model, resnet20_fuse_pairs, fuse_impl=fuse_batch_norm_inference, inplace=False
)

baseline_metrics = evaluate_vision_model(model_fused, test_dl)
params_orig = sum(p.numel() for p in model_fused.parameters())
flops_orig = count_model_flops(model_fused, (1, 3, 32, 32), formatted=False)

print(
    f"[original] Loss: {baseline_metrics['loss']:.4f}, "
    f"Acc: {baseline_metrics['accuracy']:.4f}, "
    f"Params: {params_orig}, FLOPs: {flops_orig}"
)

ratios = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
    1.00,
]

layer_keys = [k for k in get_all_convs_and_linears(model)]

print(layer_keys)
layers_to_retain = []

for layer in layers_to_retain:
    if layer in layer_keys:
        layer_keys.remove(layer)
# del layer_keys[layer_keys.index("linear")]

models_dir = Path(args.models_dir)
models_dir.mkdir(parents=True, exist_ok=True)

results = []

for k in ratios:
    model_lr = to_low_rank_activation_aware_global(
        model,
        train_dl,
        ratio_to_keep=k,
        inplace=False,
        keys=layer_keys,
        metric=args.metric,
    )
    model_eval = fuse_conv_bn(
        model_lr,
        resnet20_fuse_pairs,
        fuse_impl=fuse_batch_norm_inference,
        inplace=False,
    )

    params_lr = sum(p.numel() for p in model_eval.parameters())
    flops_raw_lr = count_model_flops(model_eval, (1, 3, 32, 32), formatted=False)
    eval_lr = evaluate_vision_model(model_eval.to(device), test_dl)

    print(
        f"[ratio={k:.2f}] Loss: {eval_lr['loss']:.4f}, "
        f"Acc: {eval_lr['accuracy']:.4f}, "
        f"Param‑keep: {params_lr/params_orig:.4f}, "
        f"FLOPs‑keep: {flops_raw_lr['total']/flops_orig['total']:.4f}"
    )

    save_path = models_dir / (
        f"{args.model_name}_reg{reg_strength:.3f}_ratio{k:.2f}_{args.metric}.pt"
        if reg_strength is not None
        else f"{args.model_name}_ratio{k:.2f}_{args.metric}.pt"
    )

    torch.save(
        {
            "model": model_lr.to("cpu"),
            "compression_ratio": k,
            "metric": args.metric,
            "reg_strength": reg_strength,
            "seed": args.seed,
        },
        save_path,
    )
    print(f"model saved to {save_path}")

    results.append(
        {
            "metric_value": k,
            "loss": eval_lr["loss"],
            "accuracy": eval_lr["accuracy"],
            "params_ratio": params_lr / params_orig,
            "flops_ratio": flops_raw_lr["total"] / flops_orig["total"],
            "metric_name": args.metric,
            "model_path": str(save_path),
            "reg_strength": reg_strength,
        }
    )

with open(args.output_file, "w") as f:
    json.dump(results, f, indent=4)
print(f"All results written to {args.output_file}")

metric_ratios = [
    res["flops_ratio"] if args.metric == "flops" else res["params_ratio"]
    for res in results
]
accuracies = [res["accuracy"] for res in results]

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.plot(metric_ratios, accuracies, marker="o")

plt.xlabel("FLOPs Ratio" if args.metric == "flops" else "Params Ratio")
plt.ylabel("Accuracy")
plt.title(f"Accuracy vs. {'FLOPs' if args.metric == 'flops' else 'Params'} Ratio")
plt.grid(True)

# Force the 90% accuracy bar to appear
plt.axhline(0.90, color="red", linestyle="--", linewidth=1.5, label="90% Accuracy")
plt.legend()

# Save the plot to a PNG file
plot_filename = f"{args.model_name}_accuracy_vs_{args.metric}.png"
plt.savefig(plot_filename, bbox_inches="tight")
plt.close()

print(f"Accuracy plot saved to {plot_filename}")


latex_file = f"{args.model_name}_results_table.tex"

with open(latex_file, "w") as f:
    f.write("\\begin{tabular}{|c|c|c|c|c|}\n")
    f.write("\\hline\n")
    f.write("Ratio & Accuracy & Loss & Params Ratio & FLOPs Ratio \\\\\n")
    f.write("\\hline\n")
    for res in results:
        f.write(
            f"{res['metric_value']:.2f} & "
            f"{res['accuracy']:.4f} & "
            f"{res['loss']:.4f} & "
            f"{res['params_ratio']:.4f} & "
            f"{res['flops_ratio']:.4f} \\\\\n"
        )
    f.write("\\hline\n")
    f.write("\\end{tabular}\n")

print(f"LaTeX table saved to {latex_file}")
