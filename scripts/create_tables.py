#!/usr/bin/env python3
import argparse
import json
import os
from typing import List, Optional, Dict

RESULTS_BASENAME = "results.json"


def load_json(path: str) -> Optional[List[Dict]]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def nearest_row(rows: List[Dict], key: str, target: float) -> Optional[Dict]:
    if not rows:
        return None

    def keyfn(r):
        if key not in r:
            return (float("inf"), 0.0)
        try:
            x = float(r[key])
        except (TypeError, ValueError):
            return (float("inf"), 0.0)
        return (abs(x - target), -x)

    return min(rows, key=keyfn)


def get_val(row: Optional[Dict], key: str) -> Optional[float]:
    if row is None or key not in row:
        return None
    try:
        return float(row[key])
    except (TypeError, ValueError):
        return None


def fmt_delta(x: Optional[float], base: Optional[float], decimals: int) -> str:
    """Compute (x-base)*100 if already percent, otherwise just (x-base)."""
    if x is None or base is None:
        return r"\textemdash{}"
    d = x - base
    return f"{d:+.{decimals}f}"


def main():
    ap = argparse.ArgumentParser(
        description="Auto-only rows with deltas relative to baseline (ratio=1.0), in percentage points."
    )
    ap.add_argument("folder", help="Folder with flops_auto and params_auto results.")
    ap.add_argument(
        "--ratios", default="0.3,0.5,0.7,1.0", help="Comma-separated ratios (0–1)"
    )
    ap.add_argument(
        "--y-key", default="accuracy", help="Metric key (default: accuracy)"
    )
    ap.add_argument("--decimals", type=int, default=2, help="Decimals for printing")
    args = ap.parse_args()

    try:
        ratios = [float(x.strip()) for x in args.ratios.split(",") if x.strip()]
    except ValueError:
        raise SystemExit("Error: --ratios must be comma-separated floats")

    def load_series(name: str) -> List[Dict]:
        p1 = os.path.join(args.folder, name, RESULTS_BASENAME)
        p2 = os.path.join(args.folder, f"{name}.json")
        return load_json(p1) or load_json(p2) or []

    flops_auto = load_series("flops_auto")
    params_auto = load_series("params_auto")

    # baselines at ratio ~ 1.0
    fa_base = nearest_row(flops_auto, "flops_ratio", 1.0)
    pa_base = nearest_row(params_auto, "params_ratio", 1.0)

    # convert baselines to percentages
    fa_base_f = get_val(fa_base, "flops_ratio") * 100 if fa_base else None
    fa_base_p = get_val(fa_base, "params_ratio") * 100 if fa_base else None
    fa_base_a = get_val(fa_base, args.y_key) * 100 if fa_base else None

    pa_base_f = get_val(pa_base, "flops_ratio") * 100 if pa_base else None
    pa_base_p = get_val(pa_base, "params_ratio") * 100 if pa_base else None
    pa_base_a = get_val(pa_base, args.y_key) * 100 if pa_base else None

    lines = []

    for idx, r in enumerate(ratios):
        # flops-auto
        fa_row = nearest_row(flops_auto, "flops_ratio", r)
        fa_f = fmt_delta(
            get_val(fa_row, "flops_ratio") * 100 if fa_row else None,
            fa_base_f,
            args.decimals,
        )
        fa_p = fmt_delta(
            get_val(fa_row, "params_ratio") * 100 if fa_row else None,
            fa_base_p,
            args.decimals,
        )
        fa_a = fmt_delta(
            get_val(fa_row, args.y_key) * 100 if fa_row else None,
            fa_base_a,
            args.decimals,
        )
        lines.append(f"FA-{r:g} & {fa_f} & {fa_p} & {fa_a} \\\\")

        # params-auto
        pa_row = nearest_row(params_auto, "params_ratio", r)
        pa_f = fmt_delta(
            get_val(pa_row, "flops_ratio") * 100 if pa_row else None,
            pa_base_f,
            args.decimals,
        )
        pa_p = fmt_delta(
            get_val(pa_row, "params_ratio") * 100 if pa_row else None,
            pa_base_p,
            args.decimals,
        )
        pa_a = fmt_delta(
            get_val(pa_row, args.y_key) * 100 if pa_row else None,
            pa_base_a,
            args.decimals,
        )
        lines.append(f"PA-{r:g} & {pa_f} & {pa_p} & {pa_a} \\\\")

        if idx != len(ratios) - 1:
            lines.append(r"\midrule")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
