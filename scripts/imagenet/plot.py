#!/usr/bin/env python3
import json
import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# Compact, publication-ready settings
# Using default Matplotlib style to avoid external dependencies
FIGSIZE = (3.5, 2.5)  # figure size in inches (width, height)
DPI = 300  # resolution in dots per inch

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "lines.linewidth": 1.0,
        "lines.markersize": 4,
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
        print(
            f"Warning: Results file not found at '{path}'. This data will be excluded from the plot."
        )
        return None
    except json.JSONDecodeError:
        print(
            f"Warning: Could not decode JSON from '{path}'. This data will be excluded from the plot."
        )
        return None


def plot_tradeoff(ax, x_vals, y_vals, label, marker, linestyle, alpha=0.9):
    """Plot a tradeoff curve on the given Axes."""
    ax.plot(
        x_vals,
        y_vals,
        marker=marker,
        label=label,
        linestyle=linestyle,
        markerfacecolor="none",  # hollow markers to reveal overlaps
        alpha=alpha,  # slight transparency
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot compact Accuracy vs FLOPs/Params for ResNet18 sweeps and save as PDF."
    )

    # ResNet18 inputs
    parser.add_argument(
        "--res18_flops", required=True, help="Path to ResNet18 flops_auto results.json"
    )
    parser.add_argument(
        "--res18_params",
        required=True,
        help="Path to ResNet18 params_auto results.json",
    )
    parser.add_argument(
        "--res18_energy", required=True, help="Path to ResNet18 energy results.json"
    )
    parser.add_argument(
        "--res18_energy_act_aware",
        required=True,
        help="Path to ResNet18 energy_act_aware results.json",
    )

    # Output directory
    parser.add_argument(
        "--output_dir", default=".", help="Directory to save the generated plots"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Organize network configurations, loading results with error handling
    networks = {
        "resnet18": {
            "flops_auto": load_results(args.res18_flops),
            "params_auto": load_results(args.res18_params),
            "energy": load_results(args.res18_energy),
            "energy_act_aware": load_results(args.res18_energy_act_aware),
        }
    }

    # Marker styles per method
    markers = {
        "flops_auto": "o",
        "params_auto": "o",
        "energy": "s",
        "energy_act_aware": "^",
    }

    # Iterate through networks and merge both metrics into a single plot with twin x-axes
    for net_name, methods in networks.items():
        fig, ax = plt.subplots(figsize=FIGSIZE)  # bottom axis: FLOPs
        ax_top = ax.twiny()  # top axis: Params
        plotted_any = False

        # Track limits separately so each x-axis fits its own data
        flops_x_all, params_x_all = [], []

        for metric in ["flops", "params"]:
            for method in [f"{metric}_auto", "energy", "energy_act_aware"]:
                results = methods.get(method)
                if results is not None:
                    x_key = f"{metric}_ratio"
                    x_vals = [r[x_key] for r in results if x_key in r]
                    y_vals = [r["accuracy"] for r in results if "accuracy" in r]

                    if x_vals and y_vals and len(x_vals) == len(y_vals):
                        linestyle = "--" if metric == "flops" else "-"
                        target_ax = ax if metric == "flops" else ax_top
                        plot_tradeoff(
                            target_ax,
                            x_vals,
                            y_vals,
                            method,
                            markers.get(method, "o"),
                            linestyle,
                            alpha=0.9,
                        )
                        if metric == "flops":
                            flops_x_all.extend(x_vals)
                        else:
                            params_x_all.extend(x_vals)
                        plotted_any = True
                    else:
                        print(
                            f"Warning: Skipping '{method}' for {net_name} {metric} plot due to missing '{x_key}' or 'accuracy' or mismatched lengths."
                        )
                else:
                    print(
                        f"Skipping '{method}' for {net_name} {metric} plot as results are unavailable."
                    )

        if not plotted_any:
            print(
                f"No data available to plot for {net_name}: merged Accuracy vs FLOPs/Params Ratio. Skipping plot generation."
            )
            plt.close(fig)
            continue

        # Set independent x-limits for each axis (so curves don't overlap awkwardly)
        if flops_x_all:
            ax.set_xlim(min(flops_x_all), max(flops_x_all))
        if params_x_all:
            ax_top.set_xlim(min(params_x_all), max(params_x_all))

        # Clean look
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        # Keep the top axis spine visible so users notice the second scale
        ax_top.spines["top"].set_visible(True)
        ax_top.spines["right"].set_visible(False)

        # Grid on y only
        ax.grid(axis="y", linestyle="--", linewidth=0.5)
        ax.yaxis.set_major_locator(MultipleLocator(0.1))

        # Labels
        ax.set(
            xlabel="FLOPs Ratio",
            ylabel="Accuracy",
        )
        ax_top.set_xlabel("Params Ratio")
        ax.set_title(f"{net_name.capitalize()}: Accuracy vs FLOPs/Params Ratio")

        # One legend pulling from both axes
        handles, labels = [], []
        for a in (ax, ax_top):
            h, l = a.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)
        # Remove duplicates while preserving order
        seen = set()
        uniq = [
            (h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))
        ]
        if uniq:
            ax.legend(*zip(*uniq), frameon=False, loc="lower right")

        plt.tight_layout(pad=0.2)
        out_file = os.path.join(
            args.output_dir, f"{net_name}_acc_vs_flops_params_merged.pdf"
        )
        fig.savefig(out_file)
        plt.close(fig)
        print(f"Saved {out_file}")
