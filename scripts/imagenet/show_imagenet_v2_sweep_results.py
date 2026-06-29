import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "scripts"))

from plot_style import (
    COLORS,
    FIGSIZE_APPENDIX,
    apply_paper_style,
    line_handle,
    paper_line_kwargs,
    save_pdf,
    style_axes,
)

apply_paper_style()


IMAGENETV2_VARIANT_PRETTY = {
    "matched_frequency": "Matched Frequency",
    "threshold0.7": "Threshold 0.7",
    "topimages": "Top Images",
}


def _load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def _index_imagenetv2_variants(data):
    indexed = {}
    entries = data.get("clean_test", {}).get("imagenetv2_calibration_variants", [])
    for entry in entries:
        name = entry.get("variant")
        variants = entry.get("variants", [])
        if name is None:
            continue
        indexed[name] = variants
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


def _plot_variant(
    variant,
    params_variants,
    flops_variants,
    params_clean_ref,
    flops_clean_ref,
    out_dir,
):
    fig, ax = plt.subplots(figsize=FIGSIZE_APPENDIX)

    p_x, p_y = _series_from_variants(params_variants)
    if p_x:
        ax.plot(
            p_x,
            p_y,
            **paper_line_kwargs(COLORS["blue"], marker="o", linestyle="-"),
        )

    f_x, f_y = _series_from_variants(flops_variants)
    if f_x:
        ax.plot(
            f_x,
            f_y,
            **paper_line_kwargs(COLORS["blue"], marker="x", linestyle="--"),
        )

    ref_p_x, ref_p_y = _series_from_variants(params_clean_ref)
    if ref_p_x:
        ax.plot(
            ref_p_x,
            ref_p_y,
            **paper_line_kwargs(COLORS["red"], marker="o", linestyle="-"),
        )

    ref_f_x, ref_f_y = _series_from_variants(flops_clean_ref)
    if ref_f_x:
        ax.plot(
            ref_f_x,
            ref_f_y,
            **paper_line_kwargs(COLORS["red"], marker="x", linestyle="--"),
        )

    variant_pretty = IMAGENETV2_VARIANT_PRETTY.get(variant, variant)
    ax.set_title(f"ImageNet-V2: {variant_pretty}", pad=3)
    ax.set_xlabel("Target keep ratio")
    ax.set_ylabel("Accuracy")
    style_axes(ax)
    if variant == "matched_frequency":
        calibration_handles = [
            line_handle(COLORS["blue"], "ImageNet-V2", marker="o"),
            line_handle(COLORS["red"], "ImageNet", marker="o"),
        ]
        target_handles = [
            line_handle(COLORS["gray"], "Params", marker="o", linestyle="-"),
            line_handle(COLORS["gray"], "FLOPs", marker="x", linestyle="--"),
        ]
        legend1 = ax.legend(
            handles=calibration_handles,
            title="Calibration data",
            loc="upper left",
            bbox_to_anchor=(0.02, 0.98),
            frameon=True,
            facecolor="white",
            edgecolor="none",
            framealpha=0.85,
            ncol=1,
            columnspacing=0.9,
            handlelength=1.8,
            handletextpad=0.35,
        )
        ax.legend(
            handles=target_handles,
            title="Compression target",
            loc="lower left",
            bbox_to_anchor=(0.02, 0.02),
            frameon=True,
            facecolor="white",
            edgecolor="none",
            framealpha=0.85,
            ncol=1,
            columnspacing=0.9,
            handlelength=1.8,
            handletextpad=0.35,
        )
        ax.add_artist(legend1)
    fig.tight_layout(pad=0.25)

    out_path = out_dir / f"{variant}.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_pdf(fig, out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--params_auto_json", required=True)
    parser.add_argument("--flops_auto_json", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    params_data = _load_json(args.params_auto_json)
    flops_data = _load_json(args.flops_auto_json)

    params_idx = _index_imagenetv2_variants(params_data)
    flops_idx = _index_imagenetv2_variants(flops_data)
    params_clean_ref = _clean_reference_variants(params_data)
    flops_clean_ref = _clean_reference_variants(flops_data)

    all_variants = sorted(set(params_idx.keys()) | set(flops_idx.keys()))
    if not all_variants:
        raise SystemExit(
            "No ImageNet-V2 calibration data found in the provided JSON files."
        )

    out_dir = Path(args.out_dir)
    for variant in all_variants:
        _plot_variant(
            variant=variant,
            params_variants=params_idx.get(variant, []),
            flops_variants=flops_idx.get(variant, []),
            params_clean_ref=params_clean_ref,
            flops_clean_ref=flops_clean_ref,
            out_dir=out_dir,
        )

    print(f"Saved {len(all_variants)} ImageNet-V2 PDF plots to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
