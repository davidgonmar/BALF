"""
This script is used to render plots in PDF format with the results of our method
given the outputs of different sweep runs (see ./cifar10/factorize_sweep.py and
./imagenet/factorize_sweep.py). It will be called individually from there, but this
Python script is shared between the two.
"""

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
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "legend.fontsize": 7,  # larger legend text
        "legend.title_fontsize": 6,  # larger legend title
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 1.0,
        "lines.markersize": 2,
        "figure.dpi": DPI,
    }
)

model_name_to_pretty_name = {
    "resnet20": "ResNet-20",
    "resnet56": "ResNet-56",
    "resnet18": "ResNet-18",
    "resnet50": "ResNet-50",
    "mobilenet_v2": "MobileNet-V2",
    "resnext50_32x4d": "ResNeXt-50 (32x4d)",
    "resnext101_32x8d": "ResNeXt-101 (32x8d)",
    "vit_b_16": "ViT-B/16",
    "deit_b_16": "DeiT-B/16",
}


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
    if not results or not isinstance(results, (list, tuple)):
        return [], []
    xs = [
        r[x_key] for r in results if isinstance(r, dict) and x_key in r and y_key in r
    ]
    ys = [
        r[y_key] for r in results if isinstance(r, dict) and x_key in r and y_key in r
    ]
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


def _pad_limits(vmin, vmax, frac=0.05):
    """Avoid singular transform when vmin == vmax."""
    if vmin == vmax:
        pad = (abs(vmin) if vmin != 0 else 1.0) * frac
        return vmin - pad, vmax + pad
    return vmin, vmax


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
        "energy_aa": load_results(args.energy_act_aware_json),
    }

    method_color = {
        "auto": "C0",
        "energy": "C1",
        "energy_aa": "C2",
    }

    # Markers per series
    markers = {
        "flops_auto": "o",
        "params_auto": "o",
        "energy": "s",
        "energy_aa": "^",
    }

    # Helper to map series key -> method bucket for colors
    def series_method(key):
        if key in ("flops_auto", "params_auto"):
            return "auto"
        if key == "energy":
            return "energy"
        if key == "energy_aa":
            return "energy_aa"
        return "auto"

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax_top = ax.twiny()

    plotted_any = False
    flops_x_all, params_x_all = [], []

    for key in ["flops_auto", "energy", "energy_aa"]:
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

    for key in ["params_auto", "energy", "energy_aa"]:
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
        raise RuntimeError("No data to plot. Exiting.")

    if flops_x_all:
        ax.set_xlim(*_pad_limits(min(flops_x_all), max(flops_x_all)))
    if params_x_all:
        ax_top.set_xlim(*_pad_limits(min(params_x_all), max(params_x_all)))

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.5)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))

    ax.set_xlabel("FLOPs Ratio")
    ax_top.set_xlabel("Params Ratio")
    ax.set_ylabel("Accuracy")
    pretty = model_name_to_pretty_name.get(args.model_name, args.model_name)
    ax.set_title(f"{pretty}")

    # Legend A: line styles (axis semantics)
    linestyle_handles = [
        Line2D([0], [0], linestyle="-", color="k", label="Params (solid)"),
        Line2D([0], [0], linestyle="--", color="k", label="FLOPs (dashed)"),
    ]

    # ResNeXt-101 has curves to the left, so move legend to right
    if args.model_name == "resnext101_32x8d":
        legend_styles = ax.legend(
            handles=linestyle_handles,
            loc="center right",  # centered to the right of the axes
            frameon=False,
            title=None,
        )
    else:
        legend_styles = ax.legend(
            handles=linestyle_handles,
            loc="upper left",
            frameon=False,
            title=None,
        )
    ax.add_artist(legend_styles)

    # Legend B: colors (methods)
    method_handles = [
        Line2D([0], [0], linestyle="-", color=method_color["auto"], label="BALF"),
        Line2D([0], [0], linestyle="-", color=method_color["energy"], label="energy"),
        Line2D(
            [0], [0], linestyle="-", color=method_color["energy_aa"], label="energy-aa"
        ),
    ]

    if not args.model_name == "mobilenet_v2":
        ax.legend(
            handles=method_handles,
            loc="lower right",
            frameon=False,
            ncol=1,
        )
    else:  # mobilenet_v2 has curves to the right
        ax.legend(
            handles=method_handles,
            loc="lower left",
            frameon=False,
            ncol=1,
        )

    plt.tight_layout(pad=0.3)
    out_file = os.path.join(
        args.output_dir, f"{args.model_name}_acc_vs_flops_params.pdf"
    )
    fig.savefig(out_file)
    plt.close(fig)
    print(f"Saved {out_file}")
