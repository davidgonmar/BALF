#!/usr/bin/env python3
import json
import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D

# Compact, publication-ready settings (no external styles)
FIGSIZE = (3.5, 2.5)  # inches (width, height)
DPI = 300
plt.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 6,  # smaller legend text
        "legend.title_fontsize": 6,  # smaller legend title
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "lines.linewidth": 1.0,
        "lines.markersize": 2,  # smaller markers
        "figure.dpi": DPI,
    }
)


def load_results(path):
    """Load JSON results from a given file path.
    Returns None if the file is not found or cannot be parsed.
    """
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: Results file not found at '{path}'. Excluding from plot.")
        return None
    except json.JSONDecodeError:
        print(f"Warning: Could not decode JSON from '{path}'. Excluding from plot.")
        return None


def extract_xy(results, x_key, y_key="accuracy"):
    xs = [r[x_key] for r in results if x_key in r and y_key in r]
    ys = [r[y_key] for r in results if x_key in r and y_key in r]
    if not xs or len(xs) != len(ys):
        return [], []
    pairs = sorted(zip(xs, ys), key=lambda t: t[0])
    x_sorted, y_sorted = zip(*pairs)
    return list(x_sorted), list(y_sorted)


def plot_tradeoff(ax, x_vals, y_vals, marker, linestyle, color, alpha=0.9, label=None):
    ax.plot(
        x_vals,
        y_vals,
        marker=marker,
        linestyle=linestyle,
        label=label,
        markerfacecolor="none",
        alpha=alpha,
        color=color,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Single-plot Accuracy vs FLOPs/Params (twin x-axes) for one model."
    )
    parser.add_argument(
        "--model_name", required=True, help="For title/filename (e.g., resnet20)."
    )
    parser.add_argument(
        "--flops_json", required=True, help="Path to flops_auto results.json"
    )
    parser.add_argument(
        "--params_json", required=True, help="Path to params_auto results.json"
    )
    parser.add_argument(
        "--energy_json", required=True, help="Path to energy results.json"
    )
    parser.add_argument(
        "--energy_act_aware_json",
        required=True,
        help="Path to energy_act_aware results.json",
    )
    parser.add_argument("--output_dir", default=".", help="Directory to save the PDF")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load results
    data = {
        "flops_auto": load_results(args.flops_json),
        "params_auto": load_results(args.params_json),
        "energy": load_results(args.energy_json),
        "act-aware": load_results(args.energy_act_aware_json),
    }

    # Method → color mapping (consistent across both axes)
    method_color = {
        "auto": "C0",
        "energy": "C1",
        "energy-aw": "C2",
    }

    # Markers per series
    markers = {
        "flops_auto": "o",
        "params_auto": "o",
        "energy": "s",
        "act-aware": "^",
    }

    # Helper to map series key -> method bucket for colors
    def series_method(key):
        if key in ("flops_auto", "params_auto"):
            return "auto"
        if key == "energy":
            return "energy"
        if key == "act-aware":
            return "energy-aw"
        return "auto"

    # Create axes: bottom (FLOPs), top (Params)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax_top = ax.twiny()

    plotted_any = False
    flops_x_all, params_x_all = [], []

    # --- FLOPs axis: ONLY flops_auto + energy + act-aware (dashed) ---
    for key in ["flops_auto", "energy", "act-aware"]:
        results = data.get(key)
        if results:
            x_vals, y_vals = extract_xy(results, "flops_ratio")
            if x_vals:
                m = series_method(key)
                plot_tradeoff(
                    ax,
                    x_vals,
                    y_vals,
                    marker=markers.get(key, "o"),
                    linestyle="--",
                    color=method_color[m],
                    label=None,
                )
                flops_x_all.extend(x_vals)
                plotted_any = True
        else:
            print(f"Skipping '{key}' on FLOPs axis: no data.")

    # --- Params axis: ONLY params_auto + energy + act-aware (solid) ---
    for key in ["params_auto", "energy", "act-aware"]:
        results = data.get(key)
        if results:
            x_vals, y_vals = extract_xy(results, "params_ratio")
            if x_vals:
                m = series_method(key)
                plot_tradeoff(
                    ax_top,
                    x_vals,
                    y_vals,
                    marker=markers.get(key, "o"),
                    linestyle="-",
                    color=method_color[m],
                    label=None,
                )
                params_x_all.extend(x_vals)
                plotted_any = True
        else:
            print(f"Skipping '{key}' on Params axis: no data.")

    if not plotted_any:
        print("No data available to plot. Exiting without saving.")
        plt.close(fig)
        raise SystemExit(0)

    # Independent x-lims for each axis
    if flops_x_all:
        ax.set_xlim(min(flops_x_all), max(flops_x_all))
    if params_x_all:
        ax_top.set_xlim(min(params_x_all), max(params_x_all))

    # Clean aesthetics
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.5)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))

    # Labels and title
    ax.set_xlabel("FLOPs Ratio")
    ax_top.set_xlabel("Params Ratio")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{args.model_name}: Accuracy vs FLOPs/Params")

    # --------- Two legends in the TOP-LEFT ----------
    # Legend A: line styles (axis semantics)
    linestyle_handles = [
        Line2D([0], [0], linestyle="-", color="k", label="Params (solid)"),
        Line2D([0], [0], linestyle="--", color="k", label="FLOPs (dashed)"),
    ]
    legend_styles = ax.legend(
        handles=linestyle_handles,
        loc="upper left",  # top-left corner
        frameon=False,
        title=None,
    )
    ax.add_artist(legend_styles)

    # Legend B: colors (method semantics)
    method_handles = [
        Line2D([0], [0], linestyle="-", color=method_color["auto"], label="auto"),
        Line2D([0], [0], linestyle="-", color=method_color["energy"], label="energy"),
        Line2D(
            [0], [0], linestyle="-", color=method_color["energy-aw"], label="energy-aw"
        ),
    ]
    ax.legend(
        handles=method_handles,
        loc="upper left",
        bbox_to_anchor=(0, 0.80),  # a bit below the first legend
        frameon=False,
        ncol=1,
        title="Method (color)",
    )

    plt.tight_layout(pad=0.2)
    out_file = os.path.join(
        args.output_dir, f"{args.model_name}_acc_vs_flops_params.pdf"
    )
    fig.savefig(out_file)
    plt.close(fig)
    print(f"Saved {out_file}")
