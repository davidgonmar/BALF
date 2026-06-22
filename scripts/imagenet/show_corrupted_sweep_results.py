import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


TITLE_FONT_SIZE = 18
AXIS_LABEL_FONT_SIZE = 15
TICK_FONT_SIZE = 13
LEGEND_FONT_SIZE = 12


MODEL_NAME_TO_PRETTY = {
    "resnet18": "ResNet-18",
    "resnet50": "ResNet-50",
    "mobilenet_v2": "MobileNet-V2",
    "resnext50_32x4d": "ResNeXt-50 (32x4d)",
    "resnext101_32x8d": "ResNeXt-101 (32x8d)",
    "vit_b_16": "ViT-B/16",
    "deit_b_16": "DeiT-B/16",
}


def _load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def _index_corrupted_variants(data):
    indexed = {}
    entries = data.get("clean_test", {}).get("corrupted_calibration_variants", [])
    for entry in entries:
        corr = entry.get("corruption")
        sev = entry.get("severity")
        if corr is None or sev is None:
            continue
        indexed.setdefault(corr, {})[int(sev)] = entry.get("variants", [])
    return indexed


def _clean_reference_variants(data):
    return data.get("clean_test", {}).get("clean_calibration_variants", [])


def _series_from_variants(variants):
    pts = []
    for variant in variants:
        if "metric_value" not in variant or "accuracy" not in variant:
            continue
        try:
            pts.append((float(variant["metric_value"]), float(variant["accuracy"])))
        except Exception:
            continue

    pts.sort(key=lambda xy: xy[0])
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return xs, ys


def _baseline_accuracy(data):
    return float(data["clean_test"]["baseline"]["accuracy"])


def _severity_color_map(severities):
    if not severities:
        return {}

    sorted_sev = sorted(severities)
    if len(sorted_sev) == 1:
        return {sorted_sev[0]: plt.cm.viridis_r(0.5)}

    n = len(sorted_sev)
    tones = [0.1 + (0.85 - 0.1) * i / (n - 1) for i in range(n)]
    return {sev: plt.cm.viridis_r(t) for sev, t in zip(sorted_sev, tones)}


def _plot_corruption(
    corr,
    params_by_sev,
    flops_by_sev,
    params_clean_ref,
    flops_clean_ref,
    baseline_acc,
    out_dir,
    pretty_model_name,
    imagenetc_subset,
):
    severities = sorted(set(params_by_sev.keys()) | set(flops_by_sev.keys()))
    if not severities:
        return

    color_by_sev = _severity_color_map(severities)
    plt.figure(figsize=(8.8, 5.6))

    for sev in severities:
        color = color_by_sev[sev]

        p_x, p_y = _series_from_variants(params_by_sev.get(sev, []))
        if p_x:
            plt.plot(
                p_x,
                p_y,
                color=color,
                linestyle="-",
                marker="o",
                linewidth=1.8,
                markersize=4.5,
                label=f"s{sev} params",
            )

        f_x, f_y = _series_from_variants(flops_by_sev.get(sev, []))
        if f_x:
            plt.plot(
                f_x,
                f_y,
                color=color,
                linestyle="--",
                marker="x",
                linewidth=1.8,
                markersize=5.0,
                label=f"s{sev} flops",
            )

    ref_p_x, ref_p_y = _series_from_variants(params_clean_ref)
    if ref_p_x:
        plt.plot(
            ref_p_x,
            ref_p_y,
            color="crimson",
            linestyle="-",
            marker="s",
            linewidth=2.2,
            markersize=5.0,
            label="clean ref params",
        )

    ref_f_x, ref_f_y = _series_from_variants(flops_clean_ref)
    if ref_f_x:
        plt.plot(
            ref_f_x,
            ref_f_y,
            color="crimson",
            linestyle="--",
            marker="D",
            linewidth=2.2,
            markersize=4.8,
            label="clean ref flops",
        )

    plt.axhline(
        baseline_acc,
        color="black",
        linestyle=":",
        linewidth=1.2,
        label="baseline",
    )

    plt.title(
        f"ImageNet-C/{imagenetc_subset}: {corr} ({pretty_model_name})",
        fontsize=TITLE_FONT_SIZE,
    )
    plt.xlabel("Target keep ratio", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.ylabel("Clean-val accuracy", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.xticks(fontsize=TICK_FONT_SIZE)
    plt.yticks(fontsize=TICK_FONT_SIZE)
    plt.grid(True, linestyle=":", linewidth=0.8, alpha=0.8)
    plt.legend(fontsize=LEGEND_FONT_SIZE, ncol=2, frameon=True)
    plt.tight_layout()

    out_path = out_dir / f"{corr}.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, format="pdf")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--imagenetc_subset", required=True)
    parser.add_argument("--params_auto_json", required=True)
    parser.add_argument("--flops_auto_json", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    params_data = _load_json(args.params_auto_json)
    flops_data = _load_json(args.flops_auto_json)

    params_idx = _index_corrupted_variants(params_data)
    flops_idx = _index_corrupted_variants(flops_data)
    params_clean_ref = _clean_reference_variants(params_data)
    flops_clean_ref = _clean_reference_variants(flops_data)

    all_corruptions = sorted(set(params_idx.keys()) | set(flops_idx.keys()))
    if not all_corruptions:
        raise SystemExit(
            "No corrupted calibration data found in the provided JSON files."
        )

    baseline_acc = _baseline_accuracy(params_data)
    out_dir = Path(args.out_dir)
    pretty_model_name = MODEL_NAME_TO_PRETTY.get(args.model_name, args.model_name)

    for corr in all_corruptions:
        _plot_corruption(
            corr=corr,
            params_by_sev=params_idx.get(corr, {}),
            flops_by_sev=flops_idx.get(corr, {}),
            params_clean_ref=params_clean_ref,
            flops_clean_ref=flops_clean_ref,
            baseline_acc=baseline_acc,
            out_dir=out_dir,
            pretty_model_name=pretty_model_name,
            imagenetc_subset=args.imagenetc_subset,
        )

    print(
        f"Saved {len(all_corruptions)} per-corruption PDF plots to: {out_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
