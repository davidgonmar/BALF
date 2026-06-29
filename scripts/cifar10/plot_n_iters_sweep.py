"""
This script plots the results of ./n_iters_sweep.py.
It generatesp plots of number of iterations vs FLOPs ratio (to measure how close the usage is to the target) and accuracy.
"""

import argparse
import json
from pathlib import Path
import sys
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from plot_style import (
    FIGSIZE_MAIN,
    RATIO_COLORS,
    apply_paper_style,
    paper_line_kwargs,
    save_pdf,
    style_axes,
)

apply_paper_style()

FIGSIZE_N_ITERS = (FIGSIZE_MAIN[0], 2.35)

NAME_MAP = {
    "resnet20": "ResNet-20",
    "resnet56": "ResNet-56",
}


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
    p.add_argument("--title_usage", default=None)
    p.add_argument("--title_accuracy", default=None)
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
            pretty = NAME_MAP.get(model, model)
            series[pretty] = (ks, flops, acc)

    if not series:
        raise SystemExit("no data")

    fig, ax = plt.subplots(figsize=FIGSIZE_N_ITERS)
    for i, (model, (ks, flops, _)) in enumerate(sorted(series.items())):
        ax.plot(
            ks,
            flops,
            label=model,
            **paper_line_kwargs(RATIO_COLORS[i % len(RATIO_COLORS)], marker="o"),
        )
    if args.title_usage:
        ax.set_title(args.title_usage, pad=3)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("FLOPs ratio")
    style_axes(ax)
    ax.legend(frameon=False)
    out1 = Path(args.out_usage)
    out1.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.25)
    save_pdf(fig, out1)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=FIGSIZE_N_ITERS)
    for i, (model, (ks, _, acc)) in enumerate(sorted(series.items())):
        ax.plot(
            ks,
            acc,
            label=model,
            **paper_line_kwargs(RATIO_COLORS[i % len(RATIO_COLORS)], marker="o"),
        )
    if args.title_accuracy:
        ax.set_title(args.title_accuracy, pad=3)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Accuracy")
    style_axes(ax)
    ax.legend(frameon=False)
    out2 = Path(args.out_accuracy)
    out2.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.25)
    save_pdf(fig, out2)
    plt.close(fig)


if __name__ == "__main__":
    main()
