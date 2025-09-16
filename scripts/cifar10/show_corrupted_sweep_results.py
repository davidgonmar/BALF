"""
Script to plot the results of the CIFAR-10-C robustness evaluation (images only).
"""

import argparse
from pathlib import Path
import json
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _mean(lst):
    return float(np.mean(lst)) if len(lst) else float("nan")


def _pct(x):
    return None if x is None else 100.0 * float(x)


def load_results(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def summarize_mode(data):
    mode = data["meta"]["mode"]
    x_key = "params_ratio" if mode == "params_auto" else "flops_ratio"

    base_clean_f = float(data["clean"]["baseline"]["accuracy"])
    base_c10c_f = _mean([float(e["baseline"]["accuracy"]) for e in data["cifar10c"]])

    clean_variants = {v["metric_value"]: v for v in data["clean"]["variants"]}
    c10c_by_mv = defaultdict(list)
    for entry in data["cifar10c"]:
        for v in entry["variants"]:
            c10c_by_mv[v["metric_value"]].append(float(v["accuracy"]))

    rows = []
    # Baseline at ratio=1.0
    rows.append(
        {
            "mode": mode,
            "metric_value": 1.0,
            "x_ratio": 1.0,
            "clean_acc_pct": _pct(base_clean_f),
            "c10c_mean_acc_pct": _pct(base_c10c_f),
        }
    )
    # Variants
    for mv, v in clean_variants.items():
        rows.append(
            {
                "mode": mode,
                "metric_value": float(mv),
                "x_ratio": float(v[x_key]),
                "clean_acc_pct": _pct(float(v["accuracy"])),
                "c10c_mean_acc_pct": _pct(_mean(c10c_by_mv[mv])),
            }
        )

    df = (
        pd.DataFrame(rows)
        .dropna(subset=["x_ratio"])
        .sort_values("x_ratio")
        .reset_index(drop=True)
    )

    # Derive deltas and gaps (percentage points)
    base = (
        df.iloc[df["x_ratio"].astype(float).argmax()]
        if (df["x_ratio"] == 1.0).sum() == 0
        else df[df["x_ratio"] == 1.0].iloc[0]
    )
    df["delta_clean_pp"] = df["clean_acc_pct"] - base["clean_acc_pct"]
    df["delta_shift_pp"] = df["c10c_mean_acc_pct"] - base["c10c_mean_acc_pct"]
    df["excess_drop_pp"] = df["delta_shift_pp"] - df["delta_clean_pp"]
    df["gap_pp"] = df["clean_acc_pct"] - df["c10c_mean_acc_pct"]
    df["baseline_gap_pp"] = float(base["clean_acc_pct"] - base["c10c_mean_acc_pct"])
    df["excess_gap_pp"] = df["gap_pp"] - df["baseline_gap_pp"]
    return df, mode, x_key


def plot_lines_with_gap_bars(df, mode, x_key, out_pdf):
    x = df["x_ratio"].astype(float).values
    clean = df["clean_acc_pct"].values
    c10c = df["c10c_mean_acc_pct"].values

    fig, ax = plt.subplots(figsize=(6.0, 4.2), dpi=150)

    # curves
    ax.plot(x, clean, marker="o", linewidth=2, label="Clean (CIFAR-10)")
    ax.plot(x, c10c, marker="s", linewidth=2, label="CIFAR-10-C (mean)")

    # connectors + gap labels
    for xi, yc, ycc in zip(x, clean, c10c):
        ax.plot([xi, xi], [ycc, yc], color="gray", linewidth=1.0, alpha=0.7)
        midy = (yc + ycc) / 2
        gap = yc - ycc
        ax.text(
            xi, midy, f"{gap:.1f}", ha="center", va="center", fontsize=8, color="black"
        )

    # axis labels
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_xlabel("Params ratio" if x_key == "params_ratio" else "FLOPs ratio")
    ax.set_ylim(0, 100)

    # add horizontal padding on x axis
    xmin, xmax = min(x), max(x)
    pad = 0.03 * (xmax - xmin)
    ax.set_xlim(xmin - pad, xmax + pad)

    # grid and legend at bottom
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--params_auto_json", required=True)
    ap.add_argument("--flops_auto_json", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df_p, mode_p, xkey_p = summarize_mode(load_results(Path(args.params_auto_json)))
    plot_lines_with_gap_bars(df_p, mode_p, xkey_p, out / "params_auto_cifar10c.pdf")

    df_f, mode_f, xkey_f = summarize_mode(load_results(Path(args.flops_auto_json)))
    plot_lines_with_gap_bars(df_f, mode_f, xkey_f, out / "flops_auto_cifar10c.pdf")


if __name__ == "__main__":
    main()
