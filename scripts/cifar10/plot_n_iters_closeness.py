#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description="Plot k (n_iters) vs flops_ratio from results.json and save to PDF"
    )
    parser.add_argument(
        "--results_dir", required=True, help="Directory containing results.json"
    )
    parser.add_argument(
        "--out", default="k_vs_flops_ratio.pdf", help="Output PDF filename (or path)"
    )
    parser.add_argument("--title", default="k vs FLOPs Ratio", help="Plot title")
    args = parser.parse_args()

    results_path = Path(args.results_dir) / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Could not find {results_path}")

    with open(results_path, "r") as f:
        data = json.load(f)

    # Extract (k, flops_ratio) and sort by k
    points = [
        (int(d.get("metric_value", d.get("try_n_iters", 0))), float(d["flops_ratio"]))
        for d in data
    ]
    points.sort(key=lambda x: x[0])

    ks = [p[0] for p in points]
    flops_ratios = [p[1] for p in points]

    plt.figure(figsize=(6, 4))
    plt.plot(ks, flops_ratios, marker="o", linewidth=1.5)
    plt.axhline(0.6, linestyle="--", linewidth=1, label="target 0.6")
    plt.title(args.title)
    plt.xlabel("k (n_iters)")
    plt.ylabel("FLOPs ratio")
    plt.grid(True, linestyle=":", linewidth=0.8)
    plt.legend()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, format="pdf")
    print(f"Saved plot to {out_path.resolve()}")


if __name__ == "__main__":
    main()
