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
from plot_style import (
    FIGSIZE_MAIN,
    METHOD_COLORS,
    apply_paper_style,
    line_handle,
    save_pdf,
    style_axes,
)

apply_paper_style()

RESULTS_BASENAME = "results.json"

model_name_to_pretty_name = {
    "resnet20": "ResNet-20",
    "resnet56": "ResNet-56",
    "resnet18": "ResNet-18",
    "resnet50": "ResNet-50",
    "mobilenet_v2": "MobileNetV2",
    "resnext50_32x4d": "ResNeXt-50 32x4d",
    "resnext101_32x8d": "ResNeXt-101 32x8d",
    "vit_b_16": "ViT-B/16",
    "deit_b_16": "DeiT-B/16",
}


def expected_seeds_for_series(series_name):
    if series_name in {"flops_auto", "params_auto", "energy_aa", "uniform_act_aware"}:
        return [0, 1, 2]
    if series_name in {"energy", "uniform"}:
        return [0]
    raise ValueError(f"Unsupported series name: {series_name}")


def load_results(path, series_name):
    if not path:
        return None

    if os.path.isdir(path):
        expected = expected_seeds_for_series(series_name)
        combined = []
        for seed in expected:
            seed_path = os.path.join(path, f"seed-{seed}", RESULTS_BASENAME)
            if not os.path.isfile(seed_path):
                raise FileNotFoundError(
                    f"Missing expected seed results for {series_name}: {seed_path}"
                )
            seed_results = load_results(seed_path, series_name)
            if seed_results is not None:
                combined.extend(seed_results)
        return combined

    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Results file not found at '{path}'.")
    except json.JSONDecodeError:
        raise ValueError(f"Could not decode JSON from '{path}'.")


def extract_xy(results, x_key, y_key="accuracy"):
    if not results:
        return [], []

    if not isinstance(results, list):
        raise TypeError("Expected results to be a list of dictionaries.")

    rows = [r for r in results if isinstance(r, dict)]
    if not rows:
        return [], []

    grouped = {}
    for r in rows:
        if x_key not in r or y_key not in r:
            continue

        try:
            x = float(r[x_key])
            y = float(r[y_key])
        except (TypeError, ValueError):
            continue

        metric_value = r.get("metric_value")
        if metric_value == "original":
            group_key = ("metric_value", 1.0)
        elif metric_value is not None:
            try:
                group_key = ("metric_value", float(metric_value))
            except (TypeError, ValueError):
                group_key = ("metric_value", str(metric_value))
        else:
            group_key = ("x", x)

        grouped.setdefault(group_key, []).append((x, y))

    if not grouped:
        return [], []

    summaries = []
    for values in grouped.values():
        xs = [x for x, _ in values]
        ys = [y for _, y in values]
        summaries.append((sum(xs) / len(xs), sum(ys) / len(ys)))

    summaries.sort(key=lambda t: t[0])
    return (
        [x for x, _ in summaries],
        [y for _, y in summaries],
    )


def plot_tradeoff(ax, x_vals, y_vals, marker, linestyle, color, alpha=1.0, label=None):
    ax.plot(
        x_vals,
        y_vals,
        marker=marker,
        linestyle=linestyle,
        label=label,
        markerfacecolor="none",
        alpha=alpha,
        color=color,
        markeredgewidth=0.8,
        zorder=3,
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
    parser.add_argument(
        "--uniform_json", required=False, help="Path to uniform results.json"
    )
    parser.add_argument(
        "--uniform_act_aware_json",
        required=False,
        help="Path to uniform_act_aware results.json",
    )
    parser.add_argument("--output_dir", default=".", help="Directory to save the PDF")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load results
    data = {
        "flops_auto": load_results(args.flops_json, "flops_auto"),
        "params_auto": load_results(args.params_json, "params_auto"),
        "energy": load_results(args.energy_json, "energy"),
        "energy_aa": load_results(args.energy_act_aware_json, "energy_aa"),
        "uniform": load_results(args.uniform_json, "uniform"),
        "uniform_act_aware": load_results(args.uniform_act_aware_json, "uniform_act_aware"),
    }

    # Markers per series
    markers = {
        "flops_auto": "o",
        "params_auto": "o",
        "energy": "s",
        "energy_aa": "^",
        "uniform": "D",
        "uniform_act_aware": "v",
    }

    # Helper to map series key -> method bucket for colors
    def series_method(key):
        if key == "flops_auto":
            return "balf_f"
        if key == "params_auto":
            return "balf_p"
        if key == "energy":
            return "energy"
        if key == "energy_aa":
            return "energy_aa"
        if key == "uniform":
            return "uniform"
        if key == "uniform_act_aware":
            return "uniform_aa"
        return "balf_f"

    fig, ax = plt.subplots(figsize=FIGSIZE_MAIN)
    ax_top = ax.twiny()

    plotted_any = False
    flops_x_all, params_x_all = [], []

    all_keys = ["flops_auto", "params_auto", "energy", "energy_aa", "uniform", "uniform_act_aware"]

    for key in all_keys:
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
                    color=METHOD_COLORS[m],
                    label=None,
                )
                flops_x_all.extend(x_vals)
                plotted_any = True
        else:
            print(f"Skipping '{key}' on FLOPs axis: no data.")

    for key in all_keys:
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
                    color=METHOD_COLORS[m],
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

    style_axes(ax)
    ax_top.spines["right"].set_visible(False)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax_top.xaxis.set_major_locator(MultipleLocator(0.2))
    ax_top.tick_params(axis="x", length=3, width=0.8)

    ax.set_xlabel("FLOPs ratio", labelpad=3)
    ax_top.set_xlabel("Parameters ratio", labelpad=4)
    ax.set_ylabel("Accuracy")
    pretty = model_name_to_pretty_name.get(args.model_name, args.model_name)
    ax.set_title(f"{pretty}", pad=3)

    # Legend A: line styles (axis semantics)
    linestyle_handles = [
        line_handle("0.1", "Parameters", linestyle="-"),
        line_handle("0.1", "FLOPs", linestyle="--"),
    ]

    if args.model_name == "resnet20":
        legend_styles = ax.legend(
            handles=linestyle_handles,
            loc="upper left",
            frameon=False,
            title=None,
        )
        ax.add_artist(legend_styles)

    # Legend B: colors (methods)
    method_handles = [
        line_handle(METHOD_COLORS["balf_f"], "BALF-F"),
        line_handle(METHOD_COLORS["balf_p"], "BALF-P"),
        line_handle(METHOD_COLORS["energy"], "Energy"),
        line_handle(METHOD_COLORS["energy_aa"], "Energy-AA"),
        line_handle(METHOD_COLORS["uniform"], "Uniform"),
        line_handle(METHOD_COLORS["uniform_aa"], "Uniform-AA"),
    ]

    if args.model_name == "resnet20":
        ax.legend(
            handles=method_handles,
            loc="lower right",
            frameon=False,
            ncol=1,
        )

    fig.tight_layout(pad=0.25)
    out_file = os.path.join(
        args.output_dir, f"{args.model_name}_acc_vs_flops_params.pdf"
    )
    save_pdf(fig, out_file)
    plt.close(fig)
    print(f"Saved {out_file}")
