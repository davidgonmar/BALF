#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys
import matplotlib.pyplot as plt


def load_results_json(d: Path):
    jf = d / "results.json"
    if not jf.exists():
        return None
    with jf.open("r") as f:
        data = json.load(f)
    return data


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
    series = {r: sorted(sa.items(), key=lambda x: x[0]) for r, sa in by_ratio.items()}
    return series


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", required=True, nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--title", default=None)
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
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.3), ncol=3, frameon=False)
    if args.title:
        plt.title(args.title)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, format="pdf", bbox_inches="tight")
    print(f"[ok] Saved plot to {args.out}")


if __name__ == "__main__":
    main()
