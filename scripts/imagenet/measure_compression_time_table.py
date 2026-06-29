"""
Renders tabular results for ./measure_compression_time.py
"""

import argparse
import json
import statistics
from pathlib import Path

SUPPORTED = [
    "resnet18",
    "resnet50",
    "mobilenet_v2",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "vit_b_16",
    "deit_b_16",
]
PRETTY = {
    "resnet18": "ResNet-18",
    "resnet50": "ResNet-50",
    "mobilenet_v2": "MobileNet-V2",
    "resnext50_32x4d": r"ResNeXt-50 (32$\times$4d)",
    "resnext101_32x8d": r"ResNeXt-101 (32$\times$8d)",
    "vit_b_16": "ViT-B/16",
    "deit_b_16": "DeiT-B/16",
}

p = argparse.ArgumentParser()
p.add_argument("--in_json", nargs="+", required=True)
p.add_argument("--out_tex", required=True)
args = p.parse_args()


def load_rows(path):
    with open(path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict) and "models" in data:
        rows = []
        for rec in data["models"]:
            t = rec.get("timings", rec)
            t["model"] = rec.get("model", t.get("model"))
            rows.append(t)
        return rows
    return data


rows = []
for path in args.in_json:
    rows.extend(load_rows(path))


def mean_std(vals):
    vals = [float(v) for v in vals]
    return statistics.mean(vals), statistics.pstdev(vals) if len(vals) > 1 else 0.0


def fmt(vals, decimals):
    m, s = mean_std(vals)
    return rf"${m:.{decimals}f}_{{\pm {s:.{decimals}f}}}$"


rows_by_name = {}
for r in rows:
    if "model" in r:
        rows_by_name.setdefault(r["model"], []).append(r)

ordered = [(n, rows_by_name[n]) for n in SUPPORTED if n in rows_by_name]

lines = []
lines.append(r"\small")
lines.append(r"\begin{tabular}{lccccccc}")
lines.append(r"\hline")
lines.append(
    r"Model & Act. & Fact.+Whit. & Solver & Replace  & Misc & Total & Peak Mem. \\"
)
lines.append(r"\hline")

for model_name, rs in ordered:
    m = PRETTY.get(model_name, model_name)
    act = [r.get("time_activation_cache", float("nan")) for r in rs]
    rep = [r.get("time_replace", float("nan")) for r in rs]
    fac = [r.get("time_factorization_and_whitening", float("nan")) for r in rs]
    sol = [r.get("time_solver", float("nan")) for r in rs]
    tot = [r.get("time_total", float("nan")) for r in rs]
    mem = [float(r.get("peak_cuda_memory_bytes", 0.0)) / (1024**3) for r in rs]
    misc = [
        float(r.get("time_total", float("nan")))
        - (
            float(r.get("time_activation_cache", float("nan")))
            + float(r.get("time_replace", float("nan")))
            + float(r.get("time_factorization_and_whitening", float("nan")))
            + float(r.get("time_solver", float("nan")))
        )
        for r in rs
    ]
    lines.append(
        f"{m} & {fmt(act, 2)} & {fmt(fac, 2)} & {fmt(sol, 2)} & {fmt(rep, 2)} & {fmt(misc, 2)} & {fmt(tot, 2)} & {fmt(mem, 2)} \\\\"
    )

lines.append(r"\hline")
lines.append(r"\end{tabular}")

Path(args.out_tex).parent.mkdir(parents=True, exist_ok=True)
with open(args.out_tex, "w") as f:
    f.write("\n".join(lines))
