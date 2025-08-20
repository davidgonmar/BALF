#!/usr/bin/env python3
import argparse
import json
import os
import glob
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# Publication‐ready defaults
FIGSIZE = (3.5, 2.5)
DPI = 300
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


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def plot_tradeoff(ax, x, y, label, marker):
    ax.plot(x, y, marker=marker, label=label)


def main():
    p = argparse.ArgumentParser(
        description="Plot Accuracy vs Params/Flops across all reg‐factorized configs"
    )
    p.add_argument("--root-dir", required=True, help=".../results/cifar10")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    models = ["resnet20", "resnet56"]
    modes = {
        "params_auto": ("params_ratio", "Params Ratio", "o"),
        "flops_auto": ("flops_ratio", "FLOPs Ratio", "s"),
    }

    os.makedirs(args.output_dir, exist_ok=True)

    for model in models:
        cfg_base = os.path.join(args.root_dir, model, "factorized_posttrain_reg")
        # find all <...>/configs/rwX_sefY_seZ
        configs = sorted(glob.glob(os.path.join(cfg_base, "*")))
        if not configs:
            print(f" No configs found for {model} in {cfg_base}")
            continue

        for mode, (x_key, xlabel, marker) in modes.items():
            fig, ax = plt.subplots(figsize=FIGSIZE)
            for cfg in configs:
                label = os.path.basename(cfg)
                resf = os.path.join(cfg, mode, "results.json")
                if not os.path.isfile(resf):
                    continue
                records = load_json(resf)
                x = [r[x_key] for r in records]
                y = [r["accuracy"] for r in records]
                plot_tradeoff(ax, x, y, label, marker)

            # Clean up plot
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            ax.grid(axis="y", linestyle="--", linewidth=0.5)
            ax.yaxis.set_major_locator(MultipleLocator(0.1))

            ax.set(
                xlabel=xlabel,
                ylabel="Accuracy",
                title=f"{model.capitalize()}: Acc vs {xlabel}",
            )
            ax.legend(frameon=False, loc="lower right")
            plt.tight_layout(pad=0.2)

            out_path = os.path.join(
                args.output_dir,
                f"{model}_acc_vs_{xlabel.replace(' ', '_').lower()}.pdf",
            )
            fig.savefig(out_path)
            plt.close(fig)
            print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
