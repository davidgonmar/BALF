import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
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

def _pretty_name(name):
    words = str(name).replace("_", " ").replace("-", " ").split()
    out = []
    for w in words:
        if w.lower() == "jpeg":
            out.append("JPEG")
        else:
            out.append(w.capitalize())
    return " ".join(out)


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


def _severity_color_map(severities):
    if not severities:
        return {}

    sorted_sev = sorted(severities)
    if len(sorted_sev) == 1:
        return {sorted_sev[0]: plt.cm.viridis_r(0.5)}

    n = len(sorted_sev)
    tones = [0.1 + (0.85 - 0.1) * i / (n - 1) for i in range(n)]
    return {sev: plt.cm.viridis_r(t) for sev, t in zip(sorted_sev, tones)}


def _show_legend_for(dataset_name, corruption):
    if dataset_name == "CIFAR-10-C":
        return corruption == "gaussian_noise"
    if dataset_name == "ImageNet-C":
        return corruption in {"fog", "gaussian_noise"}
    return False


def _legend_column_major_order(handles, ncol):
    nrows = (len(handles) + ncol - 1) // ncol
    return [
        handles[row * ncol + col]
        for col in range(ncol)
        for row in range(nrows)
        if row * ncol + col < len(handles)
    ]


def _plot_corruption(
    corr,
    params_by_sev,
    flops_by_sev,
    params_clean_ref,
    flops_clean_ref,
    out_dir,
    title,
    ylabel,
    show_legend,
):
    severities = sorted(set(params_by_sev.keys()) | set(flops_by_sev.keys()))
    if not severities:
        return

    color_by_sev = _severity_color_map(severities)
    clean_color = COLORS["red"]
    fig, ax = plt.subplots(figsize=FIGSIZE_APPENDIX)

    for sev in severities:
        color = color_by_sev[sev]

        p_x, p_y = _series_from_variants(params_by_sev.get(sev, []))
        if p_x:
            ax.plot(
                p_x,
                p_y,
                **paper_line_kwargs(color, marker="o", linestyle="-"),
            )

        f_x, f_y = _series_from_variants(flops_by_sev.get(sev, []))
        if f_x:
            ax.plot(
                f_x,
                f_y,
                **paper_line_kwargs(color, marker="x", linestyle="--"),
            )

    ref_p_x, ref_p_y = _series_from_variants(params_clean_ref)
    if ref_p_x:
        ax.plot(
            ref_p_x,
            ref_p_y,
            **paper_line_kwargs(clean_color, marker="o", linestyle="-"),
        )

    ref_f_x, ref_f_y = _series_from_variants(flops_clean_ref)
    if ref_f_x:
        ax.plot(
            ref_f_x,
            ref_f_y,
            **paper_line_kwargs(clean_color, marker="x", linestyle="--"),
        )

    ax.set_title(title, pad=3)
    ax.set_xlabel("Target keep ratio")
    ax.set_ylabel(ylabel)
    style_axes(ax)

    if show_legend:
        severity_handles = [
            line_handle(clean_color, "Baseline", marker="o"),
            *[
                line_handle(color_by_sev[sev], f"Severity {sev}", marker="o")
                for sev in severities
            ],
        ]
        semantic_handles = [
            line_handle(COLORS["gray"], "Params", marker="o", linestyle="-"),
            line_handle(COLORS["gray"], "FLOPs", marker="x", linestyle="--"),
        ]
        severity_ncol = 2
        legend1 = ax.legend(
            handles=_legend_column_major_order(severity_handles, severity_ncol),
            title="Calibration severity",
            loc="upper left",
            bbox_to_anchor=(0.02, 0.98),
            frameon=True,
            facecolor="white",
            edgecolor="none",
            framealpha=0.85,
            ncol=severity_ncol,
            columnspacing=0.8,
            handlelength=1.6,
            handletextpad=0.35,
        )
        ax.legend(
            handles=semantic_handles,
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

    out_path = out_dir / f"{corr}.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_pdf(fig, out_path)
    plt.close(fig)


def plot_corrupted_sweep_from_files(
    params_auto_json,
    flops_auto_json,
    out_dir,
    dataset_name,
    ylabel,
):
    params_data = _load_json(params_auto_json)
    flops_data = _load_json(flops_auto_json)

    params_idx = _index_corrupted_variants(params_data)
    flops_idx = _index_corrupted_variants(flops_data)
    params_clean_ref = _clean_reference_variants(params_data)
    flops_clean_ref = _clean_reference_variants(flops_data)

    all_corruptions = sorted(set(params_idx.keys()) | set(flops_idx.keys()))
    if not all_corruptions:
        raise SystemExit(
            "No corrupted calibration data found in the provided JSON files."
        )

    out_dir = Path(out_dir)
    for corr in all_corruptions:
        title = f"{dataset_name}: {_pretty_name(corr)}"

        _plot_corruption(
            corr=corr,
            params_by_sev=params_idx.get(corr, {}),
            flops_by_sev=flops_idx.get(corr, {}),
            params_clean_ref=params_clean_ref,
            flops_clean_ref=flops_clean_ref,
            out_dir=out_dir,
            title=title,
            ylabel=ylabel,
            show_legend=_show_legend_for(dataset_name, corr),
        )

    print(
        f"Saved {len(all_corruptions)} per-corruption PDF plots to: {out_dir.resolve()}"
    )


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--params_auto_json", required=True)
    parser.add_argument("--flops_auto_json", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--ylabel", required=True)
    args = parser.parse_args()

    plot_corrupted_sweep_from_files(
        params_auto_json=args.params_auto_json,
        flops_auto_json=args.flops_auto_json,
        out_dir=args.out_dir,
        dataset_name=args.dataset_name,
        ylabel=args.ylabel,
    )


if __name__ == "__main__":
    main()
