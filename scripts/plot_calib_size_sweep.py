"""
This script is used to render plots in PDF format with the results of our method
given the outputs of different calibration sizes (see ./cifar10/calib_size_sweep.py and
./imagenet/calib_size_sweep.py). It will be called individually from there, but this
Python script is shared between the two.
"""

import argparse
import json
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
from plot_style import (
    FIGSIZE_MAIN,
    RATIO_COLORS,
    apply_paper_style,
    paper_line_kwargs,
    save_pdf,
    style_axes,
)

apply_paper_style()

MODEL_NAME_TO_PRETTY = {
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

def load_results_json(d: Path):
    jf = d / "results.json"
    if not jf.exists():
        return None
    with jf.open("r") as f:
        return json.load(f)


def load_metrics_glob(d: Path):
    rows = []
    for metrics_path in d.rglob("metrics.json"):
        try:
            with metrics_path.open("r") as f:
                js = json.load(f)
            rows.append(
                {
                    "calib_size": int(js.get("calib_size")),
                    "accuracy": float(js.get("accuracy")),
                    "ratio": (
                        float(js.get("ratio")) if js.get("ratio") is not None else None
                    ),
                }
            )
        except Exception:
            continue
    return rows if rows else None


def collect_by_ratio(d: Path):
    data = load_results_json(d)
    if data is None:
        data = load_metrics_glob(d)
    if not data:
        return None
    by_ratio = {}
    for row in data:
        try:
            r = float(row["ratio"])
            s = int(row["calib_size"])
            a = float(row["accuracy"])
        except Exception:
            continue
        if r not in by_ratio:
            by_ratio[r] = {}
        if s not in by_ratio[r] or a > by_ratio[r][s]:
            by_ratio[r][s] = a
    return {r: sorted(sa.items(), key=lambda x: x[0]) for r, sa in by_ratio.items()}


def _format_calib_size(x, _):
    if x >= 1024 and abs(x / 1024 - round(x / 1024)) < 1e-6:
        return f"{int(round(x / 1024))}k"
    if x >= 1000:
        return f"{x / 1000:g}k"
    return f"{x:g}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", required=True, nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--title", default=None)
    p.add_argument("--model_name", required=True)
    args = p.parse_args()

    fig, ax = plt.subplots(figsize=FIGSIZE_MAIN)
    any_plotted = False
    calib_sizes = []

    for rd in args.results_dir:
        d = Path(rd)
        if not d.exists():
            print(f"[warn] Missing dir {d}", file=sys.stderr)
            continue
        series = collect_by_ratio(d)
        if not series:
            print(f"[warn] No metrics in {d}", file=sys.stderr)
            continue
        for i, (ratio, pts) in enumerate(
            sorted(series.items(), key=lambda kv: kv[0], reverse=True)
        ):
            xs, ys = zip(*pts)
            calib_sizes.extend(xs)
            ax.plot(
                xs,
                ys,
                label=f"ratio={ratio:g}",
                **paper_line_kwargs(RATIO_COLORS[i % len(RATIO_COLORS)], marker="o"),
            )
            any_plotted = True

    if not any_plotted:
        print("[error] No valid data found", file=sys.stderr)
        sys.exit(1)

    ax.set_xlabel("Calibration size")
    ax.set_ylabel("Accuracy")
    if calib_sizes and min(calib_sizes) > 0:
        ax.set_xlim(0, max(calib_sizes))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax.xaxis.set_major_formatter(FuncFormatter(_format_calib_size))
        ax.minorticks_off()
    style_axes(ax)
    ax.legend(loc="lower right", frameon=False, ncol=1)

    pretty = MODEL_NAME_TO_PRETTY.get(args.model_name, args.model_name)
    if args.title:
        ax.set_title(f"{pretty} - {args.title}", pad=3)
    else:
        ax.set_title(pretty, pad=3)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.25)
    save_pdf(fig, args.out)
    plt.close(fig)
    print(f"[ok] Saved plot to {args.out}")


if __name__ == "__main__":
    main()
