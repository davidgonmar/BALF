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
import matplotlib as mpl

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

mpl.rcParams.update(
    {
        "font.size": 14,  # base font size
        "axes.titlesize": 16,  # figure title
        "axes.labelsize": 14,  # x/y labels
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    }
)


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", required=True, nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--title", default=None)
    p.add_argument("--model_name", required=True)
    args = p.parse_args()

    plt.figure(figsize=(6, 4))
    any_plotted = False

    for rd in args.results_dir:
        d = Path(rd)
        if not d.exists():
            print(f"[warn] Missing dir {d}", file=sys.stderr)
            continue
        series = collect_by_ratio(d)
        if not series:
            print(f"[warn] No metrics in {d}", file=sys.stderr)
            continue
        for ratio, pts in sorted(series.items(), key=lambda kv: kv[0], reverse=True):
            xs, ys = zip(*pts)
            plt.plot(xs, ys, marker="o", label=f"ratio={ratio:g}")
            any_plotted = True

    if not any_plotted:
        print("[error] No valid data found", file=sys.stderr)
        sys.exit(1)

    plt.xlabel("Calibration size")
    plt.ylabel("Top-1 Accuracy")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="lower right", frameon=False, ncol=1)

    pretty = MODEL_NAME_TO_PRETTY.get(args.model_name, args.model_name)
    if args.title:
        plt.title(f"{pretty} - {args.title}")
    else:
        plt.title(pretty)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(pad=0.0)
    plt.savefig(args.out, format="pdf", bbox_inches="tight", pad_inches=0.0)
    print(f"[ok] Saved plot to {args.out}")


if __name__ == "__main__":
    main()
