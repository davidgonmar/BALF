"""
This script is used to render tables in LaTeX format with the results of our method
given the outputs of different sweep runs (see ./cifar10/factorize_sweep.py and
./imagenet/factorize_sweep.py). It will be called individually from there, but this
Python script is shared between the two.
"""

import argparse
import json
import os
import statistics
from typing import List, Optional, Dict

RESULTS_BASENAME = "results.json"


def load_json(path: str) -> Optional[List[Dict]]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def discover_result_files(base_dir: str, series_name: str) -> List[str]:
    candidates: List[str] = []

    direct = os.path.join(base_dir, series_name, RESULTS_BASENAME)
    if os.path.isfile(direct):
        candidates.append(direct)

    direct_json = os.path.join(base_dir, f"{series_name}.json")
    if os.path.isfile(direct_json):
        candidates.append(direct_json)

    series_dir = os.path.join(base_dir, series_name)
    if os.path.isdir(series_dir):
        seed_paths: List[str] = []
        for child in sorted(os.listdir(series_dir)):
            if child.startswith("seed-"):
                p = os.path.join(series_dir, child, RESULTS_BASENAME)
                if os.path.isfile(p):
                    seed_paths.append(p)

        if seed_paths:
            expected_seeds = expected_seeds_for_series(series_name)
            if len(seed_paths) != len(expected_seeds):
                raise RuntimeError(
                    f"Expected {len(expected_seeds)} seed result files for '{series_name}' under '{series_dir}', found {len(seed_paths)}"
                )
            return seed_paths

    return candidates


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


def fmt_delta_with_std(
    x: Optional[float],
    std: Optional[float],
    base: Optional[float],
    decimals: int,
    scale: float = 100.0,
) -> str:
    """Format a delta as mean with a compact subscript-style uncertainty."""
    if x is None or base is None:
        return r"\textemdash{}"

    d = x - base
    if std is None:
        return f"${d * scale:+.{decimals}f}$"
    return f"${d * scale:+.{decimals}f}_{{\\pm {std * scale:.{decimals}f}}}$"


def aggregate_selected_delta_stats(
    rows_per_seed: List[List[Dict]],
    select_key: str,
    target: float,
    value_key: str,
    baseline_rows_per_seed: List[Optional[Dict]],
) -> tuple[Optional[float], Optional[float]]:
    deltas: List[float] = []
    for rows, base_row in zip(rows_per_seed, baseline_rows_per_seed):
        row = nearest_row(rows, select_key, target)
        base_val = get_val(base_row, value_key) if base_row is not None else None
        val = get_val(row, value_key) if row is not None else None
        if val is None or base_val is None:
            continue
        deltas.append(val - base_val)

    if not deltas:
        return None, None

    mean_delta = statistics.mean(deltas)
    std_delta = statistics.pstdev(deltas) if len(deltas) > 1 else 0.0
    return mean_delta, std_delta


def expected_seeds_for_series(series_name: str) -> List[int]:
    if series_name in {"flops_auto", "params_auto", "energy_act_aware", "uniform_act_aware"}:
        return [0, 1, 2]
    if series_name in {"energy", "uniform"}:
        return [0]
    raise ValueError(f"Unsupported series name: {series_name}")


def ensure_expected_seed_results(base_dir: str, series_name: str) -> None:
    expected = expected_seeds_for_series(series_name)
    series_dir = os.path.join(base_dir, series_name)
    missing = []
    for seed in expected:
        seed_path = os.path.join(series_dir, f"seed-{seed}", RESULTS_BASENAME)
        if not os.path.isfile(seed_path):
            missing.append(seed)
    if missing:
        raise FileNotFoundError(
            f"Missing expected seed results for {series_name}: seed-{', seed-'.join(map(str, missing))}"
        )


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

    def load_series(name: str) -> List[List[Dict]]:
        ensure_expected_seed_results(args.folder, name)
        files = discover_result_files(args.folder, name)
        if not files:
            return []
        return [load_json(path) or [] for path in files]

    flops_auto = load_series("flops_auto")
    params_auto = load_series("params_auto")

    # baselines at ratio ~ 1.0
    fa_base_rows = [nearest_row(rows, "flops_ratio", 1.0) for rows in flops_auto]
    pa_base_rows = [nearest_row(rows, "params_ratio", 1.0) for rows in params_auto]

    lines = []

    for idx, r in enumerate(ratios):
        # flops-auto
        fa_flops_mean, fa_flops_std = aggregate_selected_delta_stats(
            flops_auto,
            "flops_ratio",
            r,
            "flops_ratio",
            fa_base_rows,
        )
        fa_params_mean, fa_params_std = aggregate_selected_delta_stats(
            flops_auto,
            "flops_ratio",
            r,
            "params_ratio",
            fa_base_rows,
        )
        fa_acc_mean, fa_acc_std = aggregate_selected_delta_stats(
            flops_auto,
            "flops_ratio",
            r,
            args.y_key,
            fa_base_rows,
        )
        lines.append(
            f"BALF-F-{r:g} & "
            f"{fmt_delta_with_std(fa_flops_mean, fa_flops_std, 0.0, args.decimals)} & "
            f"{fmt_delta_with_std(fa_params_mean, fa_params_std, 0.0, args.decimals)} & "
            f"{fmt_delta_with_std(fa_acc_mean, fa_acc_std, 0.0, args.decimals)} \\\\"
        )

        # params-auto
        pa_flops_mean, pa_flops_std = aggregate_selected_delta_stats(
            params_auto,
            "params_ratio",
            r,
            "flops_ratio",
            pa_base_rows,
        )
        pa_params_mean, pa_params_std = aggregate_selected_delta_stats(
            params_auto,
            "params_ratio",
            r,
            "params_ratio",
            pa_base_rows,
        )
        pa_acc_mean, pa_acc_std = aggregate_selected_delta_stats(
            params_auto,
            "params_ratio",
            r,
            args.y_key,
            pa_base_rows,
        )
        lines.append(
            f"BALF-P-{r:g} & "
            f"{fmt_delta_with_std(pa_flops_mean, pa_flops_std, 0.0, args.decimals)} & "
            f"{fmt_delta_with_std(pa_params_mean, pa_params_std, 0.0, args.decimals)} & "
            f"{fmt_delta_with_std(pa_acc_mean, pa_acc_std, 0.0, args.decimals)} \\\\"
        )

    print("\n".join(lines))


if __name__ == "__main__":
    main()
