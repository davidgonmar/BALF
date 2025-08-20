#!/usr/bin/env python3
"""
Run low‑rank compression experiments for different Hoyer regularisation
strengths and plot accuracy–compression curves.

Each combination of:
    • reg ∈ [reg_start … reg_end]  (default 0.001–0.004)
    • metric ∈ {"params", "flops"}
launches one call to `compress_low_rank.py`.

The single‑run script writes:
    • per‑ratio JSON metrics to <save_dir>
    • full compressed‑model *.pt files to <models_dir>
and embeds the reg strength inside every *.pt file.

Plots of the family of curves are produced after all runs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt

plt.rcParams["figure.autolayout"] = True  # avoid clip‑offs
# ---------------------------------------------------------------- helpers


def run_once(
    base_script: str,
    pretrained_path: str,
    metric: str,
    output_file: Path,
    models_dir: Path,
) -> None:
    """Launch a single experiment unless its JSON cache already exists."""

    cmd = [
        "python",
        base_script,
        "--pretrained_path",
        pretrained_path,
        "--metric",
        metric,
        "--output_file",
        str(output_file),
        "--models_dir",
        str(models_dir),
    ]
    print("↳", " ".join(cmd))
    subprocess.run(cmd, check=True)


def plot_family(
    results_dir: Path,
    reg_values: List[float],
    metric_name: str,
    x_key: str,
    outfile: Path,
) -> None:
    """Plot accuracy as a function of *x_key* (params_ratio or flops_ratio)."""
    plt.figure()
    for reg in reg_values:
        fp = results_dir / f"results_reg{reg:.3f}_{metric_name}.json"
        with open(fp) as f:
            data = json.load(f)
        xs = [d[x_key] for d in data]
        ys = [d["accuracy"] for d in data]
        plt.plot(xs, ys, marker="o", label=f"reg={reg:.3f}")
    plt.xlabel(x_key.replace("_", " ").title())
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy vs {x_key.replace('_', ' ').title()}")
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False)
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ saved", outfile)


def frange(start: float, stop: float, step: float):
    """Inclusive floating range helper."""
    while start <= stop + 1e-12:
        yield round(start, 3)
        start += step


# ---------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--base_script", default="factorize3.py", help="Path to the single‑run script."
    )
    parser.add_argument(
        "--save_dir",
        default="compression_runs",
        help="Where the per‑ratio JSON files & plots go.",
    )
    parser.add_argument(
        "--models_dir",
        default="compressed_models",
        help="Root directory in which *.pt models are stored.",
    )
    parser.add_argument("--reg_start", type=float, default=0.001)
    parser.add_argument("--reg_end", type=float, default=0.004)
    parser.add_argument("--reg_step", type=float, default=0.001)
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    reg_vals = list(frange(args.reg_start, args.reg_end, args.reg_step))
    metrics = ["params", "flops"]

    # ---------------- run experiments ----------------
    print("=== Running low‑rank compression experiments ===")
    for reg in reg_vals:
        ckpt = f"./results/cifar10_resnet20_hoyer_finetuned_reg{reg:.3f}.pth"
        for metric in metrics:
            json_out = save_dir / f"results_reg{reg:.3f}_{metric}.json"
            subdir = models_dir / f"reg{reg:.3f}_{metric}"
            subdir.mkdir(parents=True, exist_ok=True)
            run_once(args.base_script, ckpt, metric, json_out, subdir)

    # ---------------- plotting ----------------
    print("\n=== Plotting curves ===")
    plot_family(
        save_dir, reg_vals, "params", "params_ratio", save_dir / "curves_params.png"
    )
    plot_family(
        save_dir, reg_vals, "flops", "flops_ratio", save_dir / "curves_flops.png"
    )
    print("All done ✨")


if __name__ == "__main__":
    main()
