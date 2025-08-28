#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt


def load_points(results_path: Path):
    with open(results_path, "r") as f:
        data = json.load(f)

    # Extract (k, flops_ratio) and sort by k
    points = [
        (int(d.get("metric_value", d.get("try_n_iters", 0))), float(d["flops_ratio"]))
        for d in data
        if "flops_ratio" in d
    ]
    points.sort(key=lambda x: x[0])
    ks = [p[0] for p in points]
    flops_ratios = [p[1] for p in points]
    return ks, flops_ratios


def main():
    parser = argparse.ArgumentParser(
        description="Plot k (n_iters) vs flops_ratio for multiple models on one PDF"
    )
    parser.add_argument(
        "--results_dir",
        required=True,
        help="Directory containing per-model subfolders with results.json (e.g., resnet20/results.json)",
    )
    parser.add_argument(
        "--out",
        default="k_vs_flops_ratio_multi.pdf",
        help="Output PDF filename (or path)",
    )
    parser.add_argument(
        "--title", default="k vs FLOPs Ratio (all models)", help="Plot title"
    )
    parser.add_argument(
        "--target",
        type=float,
        default=0.6,
        help="Horizontal reference line for target FLOPs ratio",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        help="Optional subset of model subfolder names to include (defaults to all found).",
    )
    args = parser.parse_args()

    root = Path(args.results_dir)
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")

    # Discover candidate model subdirs that have results.json
    candidates = []
    for p in sorted(root.iterdir()):
        if p.is_dir() and (p / "results.json").exists():
            candidates.append(p)

    if args.models:
        name_set = set(args.models)
        candidates = [p for p in candidates if p.name in name_set]

    if not candidates:
        raise FileNotFoundError(
            f"No per-model results.json files found in {root} "
            f"(looked for <model>/results.json)."
        )

    plt.figure(figsize=(7.5, 4.5))

    any_plotted = False
    for model_dir in candidates:
        results_path = model_dir / "results.json"
        try:
            ks, flops = load_points(results_path)
            if not ks:
                print(f"Warning: no data points in {results_path}, skipping.")
                continue
            plt.plot(ks, flops, marker="o", linewidth=1.5, label=model_dir.name)
            any_plotted = True
        except Exception as e:
            print(f"Failed to load {results_path}: {e}")

    if not any_plotted:
        raise RuntimeError("No valid series to plot.")

    # Reference target line
    if args.target is not None:
        plt.axhline(
            args.target, linestyle="--", linewidth=1, label=f"target {args.target}"
        )

    plt.title(args.title)
    plt.xlabel("k (n_iters)")
    plt.ylabel("FLOPs ratio")
    plt.grid(True, linestyle=":", linewidth=0.8)
    plt.legend()
    plt.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, format="pdf")
    print(f"Saved plot to {out_path.resolve()}")


if __name__ == "__main__":
    main()
