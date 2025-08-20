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


def plot_tradeoff(ax, x_vals, y_vals, label, marker):
    """Plot a tradeoff curve on the given Axes."""
    ax.plot(x_vals, y_vals, marker=marker, label=label)


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

    # Iterate through networks and metrics (flops vs params)
    for net_name, methods in networks.items():
        for metric in ["flops", "params"]:
            fig, ax = plt.subplots(figsize=FIGSIZE)

            plotted_any = False  # Flag to check if any data was plotted for this metric

            for method in [f"{metric}_auto", "energy", "energy_act_aware"]:
                # Ensure correct key names for params vs flops key
                # The 'method' variable already holds the correct key name, e.g., 'flops_auto'
                results = methods.get(method)  # Use .get() for safe access

                if results is not None:
                    x_key = f"{metric}_ratio"
                    x_vals = [
                        r[x_key] for r in results if x_key in r
                    ]  # Ensure key exists in individual result dict
                    y_vals = [
                        r["accuracy"] for r in results if "accuracy" in r
                    ]  # Ensure key exists

                    if x_vals and y_vals and len(x_vals) == len(y_vals):
                        plot_tradeoff(
                            ax, x_vals, y_vals, method, markers.get(method, "o")
                        )  # Fallback marker
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
                    f"No data available to plot for {net_name}: Accuracy vs {metric.capitalize()} Ratio. Skipping plot generation."
                )
                plt.close(fig)  # Close the empty figure
                continue  # Move to the next metric/network

            # Remove top/right spines for a cleaner look
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)

            # Add light dashed grid lines on y-axis
            ax.grid(axis="y", linestyle="--", linewidth=0.5)
            ax.yaxis.set_major_locator(MultipleLocator(0.1))  # Adjust y-axis ticks

            # Labeling
            xlabel = "FLOPs Ratio" if metric == "flops" else "Params Ratio"
            ax.set(
                xlabel=xlabel,
                ylabel="Accuracy",
                title=f"{net_name.capitalize()}: Accuracy vs {xlabel}",
            )
            ax.legend(frameon=False, loc="lower right")
            plt.tight_layout(pad=0.2)  # Tight layout for compact figures

            # Save PDF
            out_file = os.path.join(args.output_dir, f"{net_name}_acc_vs_{metric}.pdf")
            fig.savefig(out_file)
            plt.close(fig)  # Close the figure to free memory
            print(f"Saved {out_file}")
