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
    """Load JSON results from a given file path."""
    with open(path, "r") as f:
        return json.load(f)


def plot_tradeoff(ax, x_vals, y_vals, label, marker):
    """Plot a tradeoff curve on the given Axes."""
    ax.plot(x_vals, y_vals, marker=marker, label=label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot compact Accuracy vs FLOPs/Params for ResNet sweeps and save as PDF."
    )
    # ResNet20 inputs
    parser.add_argument(
        "--res20_flops", required=True, help="Path to ResNet20 flops_auto results.json"
    )
    parser.add_argument(
        "--res20_params",
        required=True,
        help="Path to ResNet20 params_auto results.json",
    )
    parser.add_argument(
        "--res20_energy", required=True, help="Path to ResNet20 energy results.json"
    )
    parser.add_argument(
        "--res20_energy_act_aware",
        required=True,
        help="Path to ResNet20 energy_act_aware results.json",
    )
    # ResNet56 inputs
    parser.add_argument(
        "--res56_flops", required=True, help="Path to ResNet56 flops_auto results.json"
    )
    parser.add_argument(
        "--res56_params",
        required=True,
        help="Path to ResNet56 params_auto results.json",
    )
    parser.add_argument(
        "--res56_energy", required=True, help="Path to ResNet56 energy results.json"
    )
    parser.add_argument(
        "--res56_energy_act_aware",
        required=True,
        help="Path to ResNet56 energy_act_aware results.json",
    )
    # Output directory
    parser.add_argument(
        "--output_dir", default=".", help="Directory to save the generated plots"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Organize network configurations
    networks = {
        "resnet20": {
            "flops_auto": load_results(args.res20_flops),
            "params_auto": load_results(args.res20_params),
            "energy": load_results(args.res20_energy),
            "energy_act_aware": load_results(args.res20_energy_act_aware),
        },
        "resnet56": {
            "flops_auto": load_results(args.res56_flops),
            "params_auto": load_results(args.res56_params),
            "energy": load_results(args.res56_energy),
            "energy_act_aware": load_results(args.res56_energy_act_aware),
        },
    }

    # Marker styles per method
    markers = {
        "flops_auto": "o",
        "params_auto": "o",
        "energy": "s",
        "energy_act_aware": "^",
    }

    # Iterate through networks and metrics (flops vs params)
    for net_name, methods in networks.items():
        for metric in ["flops", "params"]:
            fig, ax = plt.subplots(figsize=FIGSIZE)
            for method in [f"{metric}_auto", "energy", "energy_act_aware"]:
                # Ensure correct key names for params vs flops
                key = method
                results = methods[key]
                x_key = f"{metric}_ratio"
                x_vals = [r[x_key] for r in results]
                y_vals = [r["accuracy"] for r in results]
                plot_tradeoff(ax, x_vals, y_vals, key, markers[key])

            # Remove top/right spines for a cleaner look
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            # Add light dashed grid lines on y-axis
            ax.grid(axis="y", linestyle="--", linewidth=0.5)
            ax.yaxis.set_major_locator(MultipleLocator(0.1))

            # Labeling
            xlabel = "FLOPs Ratio" if metric == "flops" else "Params Ratio"
            ax.set(
                xlabel=xlabel,
                ylabel="Accuracy",
                title=f"{net_name.capitalize()}: Accuracy vs {xlabel}",
            )
            ax.legend(frameon=False, loc="lower right")
            plt.tight_layout(pad=0.2)

            # Save PDF
            out_file = os.path.join(args.output_dir, f"{net_name}_acc_vs_{metric}.pdf")
            fig.savefig(out_file)
            plt.close(fig)
            print(f"Saved {out_file}")
