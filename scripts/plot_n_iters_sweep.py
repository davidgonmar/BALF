"""
This script reads a JSON summary file containing results of various models
and generates plots showing the relationship between the number of iterations (k)
and both FLOPs ratio and accuracy.
"""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt


def extract_series(results_list):
    pts = []
    for d in results_list:
        if "flops_ratio" in d and "accuracy" in d:
            k = d.get("metric_value", d.get("try_n_iters", None))
            if k is None:
                continue
            try:
                k = int(k)
                fr = float(d["flops_ratio"])
                acc = float(d["accuracy"])
                pts.append((k, fr, acc))
            except:
                pass
    pts.sort(key=lambda x: x[0])
    ks = [p[0] for p in pts]
    flops = [p[1] for p in pts]
    acc = [p[2] for p in pts]
    return ks, flops, acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary", required=True)
    p.add_argument("--out_usage", default="k_vs_flops_ratio.pdf")
    p.add_argument("--out_accuracy", default="k_vs_accuracy.pdf")
    p.add_argument("--title_usage", default="k vs FLOPs Ratio")
    p.add_argument("--title_accuracy", default="k vs Accuracy")
    p.add_argument("--models", nargs="*")
    args = p.parse_args()

    with open(args.summary, "r") as f:
        data = json.load(f)

    if args.models:
        data = {k: v for k, v in data.items() if k in set(args.models)}

    series = {}
    for model, lst in data.items():
        ks, flops, acc = extract_series(lst)
        if ks:
            series[model] = (ks, flops, acc)

    if not series:
        raise SystemExit("no data")

    plt.figure(figsize=(7.5, 4.5))
    for model, (ks, flops, _) in sorted(series.items()):
        plt.plot(ks, flops, marker="o", linewidth=1.5, label=model)
    plt.title(args.title_usage)
    plt.xlabel("k (n_iters)")
    plt.ylabel("FLOPs ratio")
    plt.grid(True, linestyle=":", linewidth=0.8)
    plt.legend()
    out1 = Path(args.out_usage)
    out1.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out1, format=out1.suffix.lstrip(".") or None)

    plt.figure(figsize=(7.5, 4.5))
    for model, (ks, _, acc) in sorted(series.items()):
        plt.plot(ks, acc, marker="o", linewidth=1.5, label=model)
    plt.title(args.title_accuracy)
    plt.xlabel("k (n_iters)")
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle=":", linewidth=0.8)
    plt.legend()
    out2 = Path(args.out_accuracy)
    out2.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out2, format=out2.suffix.lstrip(".") or None)


if __name__ == "__main__":
    main()
