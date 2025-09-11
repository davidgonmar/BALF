#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys
import matplotlib.pyplot as plt


def load_points_from_results_json(d: Path):
    jf = d / "results.json"
    if not jf.exists():
        return None
    with jf.open("r") as f:
        data = json.load(f)
    pts = []
    for row in data:
        try:
            pts.append((int(row["calib_size"]), float(row["accuracy"])))
        except Exception:
            continue
    return pts


def load_points_from_metrics_glob(d: Path):
    pts = []
    for metrics_path in d.rglob("metrics.json"):
        try:
            with metrics_path.open("r") as f:
                js = json.load(f)
            size = int(js.get("calib_size"))
            acc = float(js.get("accuracy"))
            pts.append((size, acc))
        except Exception:
            continue
    return pts


def collect_points(d: Path):
    pts = load_points_from_results_json(d)
    if not pts:
        pts = load_points_from_metrics_glob(d)
    best = {}
    for s, a in pts:
        if s not in best or a > best[s]:
            best[s] = a
    return sorted(best.items(), key=lambda x: x[0])


def maybe_read_meta(d: Path):
    jf = d / "results.json"
    if jf.exists():
        try:
            with jf.open("r") as f:
                arr = json.load(f)
            if isinstance(arr, list) and arr:
                return {
                    "model_name": arr[0].get("model_name"),
                    "mode": arr[0].get("mode"),
                    "ratio": arr[0].get("ratio"),
                }
        except Exception:
            pass
    for metrics_path in d.rglob("metrics.json"):
        try:
            with metrics_path.open("r") as f:
                js = json.load(f)
            return {
                "model_name": js.get("model_name"),
                "mode": js.get("mode"),
                "ratio": js.get("ratio"),
            }
        except Exception:
            continue
    return None


def infer_label(d: Path, meta: dict | None) -> str:
    if not meta:
        return d.name
    parts = []
    if meta.get("model_name"):
        parts.append(meta["model_name"])
    if meta.get("mode"):
        parts.append(meta["mode"])
    if "ratio" in meta and meta["ratio"] is not None:
        parts.append(f"r={meta['ratio']}")
    return " • ".join(parts) if parts else d.name


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--results_dir", required=True, nargs="+", help="Results directories"
    )
    p.add_argument("--out", required=True, help="Output PDF file (e.g., plot.pdf)")
    p.add_argument("--title", default=None, help="Optional plot title")
    args = p.parse_args()

    plt.figure(figsize=(6, 4))
    any_plotted = False

    for rd in args.results_dir:
        d = Path(rd)
        if not d.exists():
            print(f"[warn] Missing dir {d}", file=sys.stderr)
            continue

        pts = collect_points(d)
        if not pts:
            print(f"[warn] No metrics in {d}", file=sys.stderr)
            continue

        xs, ys = zip(*pts)
        meta = maybe_read_meta(d)
        label = infer_label(d, meta)
        plt.plot(xs, ys, marker="o", label=label)
        any_plotted = True

    if not any_plotted:
        print("[error] No valid data found", file=sys.stderr)
        sys.exit(1)

    plt.xlabel("Calibration size")
    plt.ylabel("Top-1 Accuracy")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    if args.title:
        plt.title(args.title)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, format="pdf")
    print(f"[ok] Saved plot to {args.out}")


if __name__ == "__main__":
    main()
